using System.Diagnostics.CodeAnalysis;
using SkiaSharp;

namespace PgsToSrtPlus;

/// <summary>
/// Converts a raw PGS RGBA bitmap into a standardized binary grayscale image for OCR.
///
/// Pipeline:
///   1. Determine subtitle support (which pixels belong to the subtitle, not empty background)
///      — alpha-based if the channel varies; border-median fallback otherwise
///   2. Separate bright text fill from dark outline/shadow: Otsu on val=max(R,G,B),
///      then constrained region growing (fill_strong seeds + 1px dilation)
///   3. Recover thin punctuation: dilate(fill, 1) clamped back to support
///   4. Binary output: dark text on white background
///   5. Scale to a target height bucket (keep aspect ratio, Mitchell cubic)
///   6. Uniform padding
///
/// Optional debug intermediates (when DebugDir is set):
///   {prefix}_raw_rgba.png, _cropped.png, _mask.png, _text_only.png, _final_ocr_input.png
/// </summary>
static class Preprocessor
{
    public record Options
    {
        /// <summary>Alpha threshold for support detection; excludes near-transparent fringe pixels.</summary>
        public byte AlphaThreshold => 5;

        /// <summary>
        /// If max(alpha) - min(alpha) is below this, the alpha channel is considered flat
        /// (e.g. always 255) and background estimation is used for support instead.
        /// </summary>
        public byte AlphaFlatness => 10;

        /// <summary>
        /// For the no-alpha fallback: a border pixel is "background" by definition;
        /// support = |gray - bg_median| > this value.
        /// </summary>
        public byte BackgroundEps => 20;

        /// <summary>Width of the border ring used for background color estimation.</summary>
        public int BorderRingWidth => 2;

        /// <summary>Margin fraction of text height added around the tight bbox on each side.</summary>
        public float MarginFraction => 0.10f;

        /// <summary>Minimum margin in pixels, applied on each side regardless of text height.</summary>
        public int MinMarginPx => 6;

        /// <summary>If set, saves intermediate PNG files to this directory.</summary>
        public string? DebugDir { get; init; }

        /// <summary>Filename prefix for debug intermediates (e.g. "00001").</summary>
        public string? DebugPrefix { get; init; }

        /// <summary>
        /// Gaussian blur sigma applied to the binary fill mask to produce antialiased edges.
        /// 0.5 gives a 2-pixel soft edge — enough to smooth PGS outline artefacts without
        /// making text look noticeably blurry.
        /// </summary>
        public float BlurSigma => 0.5f;
    }

    // Target heights for scale normalisation.
    // Thresholds are tuned for Blu-ray PGS at ~60 px/line native height.
    const int BucketSmall  = 80;   // crop h <  90 → 80  (single line, ~60 px native)
    const int BucketMedium = 96;   // crop h < 140 → 96  (tall / two short lines)
    const int BucketLarge  = 160;  // crop h >= 140 → 160 (two full lines)
    const float PadFraction = 0.15f;

    /// <summary>
    /// Runs the preprocessing pipeline on a raw PGS RGBA bitmap.
    /// Returns a Gray8 SKBitmap (dark text on white) or null if no visible content was found.
    /// Caller owns the returned bitmap.
    /// </summary>
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public static SKBitmap? Preprocess(SKBitmap rgba, Options? opts = null)
    {
        opts ??= new Options();
        int w = rgba.Width, h = rgba.Height, n = w * h;

        // Unpack BGRA pixels (straight alpha despite SKAlphaType.Premul declaration in PgsDecoder)
        byte[] R = new byte[n], G = new byte[n], B = new byte[n], A = new byte[n];
        unsafe
        {
            byte* ptr = (byte*)rgba.GetPixels();
            for (int i = 0; i < n; i++)
            {
                B[i] = ptr[i * 4];
                G[i] = ptr[i * 4 + 1];
                R[i] = ptr[i * 4 + 2];
                A[i] = ptr[i * 4 + 3];
            }
        }

        SaveDebugRgba(rgba, opts, "raw_rgba");

        // ── Step 1: Subtitle support ──────────────────────────────────────────────────
        bool[] support = BuildSupport(R, G, B, A, w, h, opts);

        if (!TryBBox(support, w, h, out int ax0, out int ay0, out int ax1, out int ay1))
            return null;

        // Expand bbox by margin, crop all arrays
        int textH  = ay1 - ay0 + 1;
        int margin = Math.Max(opts.MinMarginPx, (int)MathF.Round(opts.MarginFraction * textH));
        int cx0 = Math.Max(0,     ax0 - margin);
        int cy0 = Math.Max(0,     ay0 - margin);
        int cx1 = Math.Min(w - 1, ax1 + margin);
        int cy1 = Math.Min(h - 1, ay1 + margin);
        int cw = cx1 - cx0 + 1, ch = cy1 - cy0 + 1;

        bool[] cs = CropBool(support, w, cx0, cy0, cw, ch);
        byte[] cR = CropByte(R, w, cx0, cy0, cw, ch);
        byte[] cG = CropByte(G, w, cx0, cy0, cw, ch);
        byte[] cB = CropByte(B, w, cx0, cy0, cw, ch);
        SaveDebugRgbaCrop(rgba, cx0, cy0, cw, ch, opts, "cropped");
        SaveDebugMask(cs, cw, ch, opts, "mask");

        // ── Step 2: Separate fill from outline/shadow via Otsu + constrained growing ──
        // val = max(R,G,B) is the brightness discriminator.
        // Otsu splits the support histogram into dark shadow and bright text fill.
        // Plain Otsu at the boundary includes shadow tendrils: pixels just above the
        // threshold that are connected to the letter via edge gradients.
        //
        // Constrained region growing avoids this:
        //   fill_strong = support pixels above 80% of the bright-class mean (bright cores)
        //   fill_weak   = support pixels above otsuT (all candidates)
        //   fill        = DilateBox(fill_strong, r=1) & fill_weak
        //
        // Tendrils connect to the letter through low-brightness edge pixels, not directly
        // to the bright core. A 1px dilation from fill_strong reaches letter edges but
        // NOT tendrils that are 2+ px from any fill_strong pixel.
        byte[] val = new byte[cw * ch];
        var supportVals = new List<byte>(cw * ch / 4);
        for (int i = 0; i < cw * ch; i++)
        {
            val[i] = Math.Max(cR[i], Math.Max(cG[i], cB[i]));
            if (cs[i]) supportVals.Add(val[i]);
        }

        if (supportVals.Count == 0)
            return null;

        byte otsuT = OtsuThreshold(supportVals);

        double sumBright = 0; int cntBright = 0;
        foreach (var v in supportVals)
            if (v >= otsuT) { sumBright += v; cntBright++; }
        byte strongT = cntBright > 0
            ? (byte)(otsuT + 0.80 * (sumBright / cntBright - otsuT))
            : otsuT;

        bool[] fillWeak   = new bool[cw * ch];
        bool[] fillStrong = new bool[cw * ch];
        for (int i = 0; i < cw * ch; i++)
        {
            fillWeak[i]   = cs[i] && val[i] >= otsuT;
            fillStrong[i] = cs[i] && val[i] >= strongT;
        }

        bool[] expanded = DilateBox(fillStrong, cw, ch, 1);
        bool[] fill     = new bool[cw * ch];
        for (int i = 0; i < cw * ch; i++)
            fill[i] = expanded[i] && fillWeak[i];

        // ── Step 3: Remove specks ─────────────────────────────────────────────────────
        // Discard tiny isolated components (noise, shadow dust Otsu let through).
        // Relative threshold: 0.01% of crop keeps i-dots and thin strokes safe.
        // Absolute floor of 3 catches sub-pixel noise on very small crops.
        int minArea = Math.Max(3, (int)(0.0001f * cw * ch));
        fill = RemoveSmallComponents(fill, cw, ch, minArea);

        SaveDebugMask(fill, cw, ch, opts, "text_only");

        // ── Step 4: Soften edges ──────────────────────────────────────────────────────
        // Blur the binary fill with sigma=1.0 to produce genuine antialiased edges.
        // At sigma=1.0 the kernel half-width is 3 px, so a 2 px stroke center stays
        // near gray=29 (nearly black) while 1 px outside the edge reaches gray=178 —
        // the gradual gradient that TrOCR expects from naturally-rendered text.
        // The contrast stretch is intentionally omitted: it pushed mid-tones toward 1
        // and counteracted the softening we want here.
        float[] f = new float[cw * ch];
        for (int i = 0; i < cw * ch; i++) f[i] = fill[i] ? 1.0f : 0.0f;

        f = GaussianBlur(f, cw, ch, sigma: opts.BlurSigma);

        // Dark text on white background: out = 255 - (f * 255)
        byte[] gray = new byte[cw * ch];
        for (int i = 0; i < cw * ch; i++)
            gray[i] = (byte)(255 - (int)(f[i] * 255.0f));

        // ── Step 5: Scale to target height ───────────────────────────────────────────
        int targetH = BucketHeight(ch);
        int targetW = Math.Max(1, (int)MathF.Round(cw * ((float)targetH / ch)));
        byte[] scaledGray = ScaleGray(gray, cw, ch, targetW, targetH);

        // ── Step 6: Padding ───────────────────────────────────────────────────────────
        // Top and bottom padding target ≈ 15% of scaled text height.
        // Floor of 14 px ensures small bucket sizes still get adequate margin.
        int padPx  = Math.Max(14, (int)MathF.Round(PadFraction * targetH));
        int finalW = targetW + 2 * padPx;
        int finalH = targetH + 2 * padPx;
        byte[] finalGray = new byte[finalW * finalH];
        Array.Fill(finalGray, (byte)255);
        for (int row = 0; row < targetH; row++)
            Array.Copy(scaledGray, row * targetW, finalGray, (row + padPx) * finalW + padPx, targetW);

        var result = MakeGray8(finalGray, finalW, finalH);
        SaveDebugGray(finalGray, finalW, finalH, opts, "final_ocr_input");
        return result;
    }

    // ── support detection ─────────────────────────────────────────────────────────────

    [SuppressMessage("ReSharper", "InconsistentNaming")]
    static bool[] BuildSupport(byte[] R, byte[] G, byte[] B, byte[] A,
        int w, int h, Options opts)
    {
        int n = w * h;
        byte alphaMin = A[0], alphaMax = A[0];
        for (int i = 1; i < n; i++)
        {
            if (A[i] < alphaMin) alphaMin = A[i];
            if (A[i] > alphaMax) alphaMax = A[i];
        }

        // Alpha varies meaningfully → use it directly
        if (alphaMax - alphaMin >= opts.AlphaFlatness)
        {
            bool[] s = new bool[n];
            for (int i = 0; i < n; i++)
                s[i] = A[i] > opts.AlphaThreshold;
            return s;
        }

        // Alpha is flat (e.g. always 255) → estimate background from border ring
        var borderGrays = new List<byte>(2 * (w + h) * opts.BorderRingWidth);
        for (int ring = 0; ring < opts.BorderRingWidth; ring++)
        {
            for (int x = 0; x < w; x++)
            {
                int top = ring * w + x;
                int bot = (h - 1 - ring) * w + x;
                borderGrays.Add(Luma(R[top], G[top], B[top]));
                borderGrays.Add(Luma(R[bot], G[bot], B[bot]));
            }
            for (int y = ring + 1; y < h - 1 - ring; y++)
            {
                int left  = y * w + ring;
                int right = y * w + (w - 1 - ring);
                borderGrays.Add(Luma(R[left],  G[left],  B[left]));
                borderGrays.Add(Luma(R[right], G[right], B[right]));
            }
        }
        borderGrays.Sort();
        byte bgGray = borderGrays[borderGrays.Count / 2]; // median

        bool[] sup = new bool[n];
        for (int i = 0; i < n; i++)
            sup[i] = Math.Abs(Luma(R[i], G[i], B[i]) - bgGray) > opts.BackgroundEps;
        return sup;
    }

    // ── Otsu threshold ────────────────────────────────────────────────────────────────

    static byte OtsuThreshold(List<byte> vals)
    {
        int[] hist = new int[256];
        foreach (var v in vals) hist[v]++;

        int total = vals.Count;
        long sum = 0;
        for (int i = 0; i < 256; i++) sum += (long)i * hist[i];

        long sumB = 0;
        int wB = 0;
        double maxVar = -1;
        byte threshold = 128;

        for (int t = 0; t < 256; t++)
        {
            wB += hist[t];
            if (wB == 0) continue;
            int wF = total - wB;
            if (wF == 0) break;

            sumB += (long)t * hist[t];
            double mB = (double)sumB / wB;
            double mF = (double)(sum - sumB) / wF;
            double between = (double)wB * wF * (mB - mF) * (mB - mF);

            if (between > maxVar)
            {
                maxVar    = between;
                threshold = (byte)t;
            }
        }
        return threshold;
    }

    // ── connected component filter ────────────────────────────────────────────────────

    /// <summary>
    /// Removes 8-connected foreground components whose pixel area is below minArea.
    /// Implemented as BFS over all set pixels; each pixel is visited at most once.
    /// </summary>
    static bool[] RemoveSmallComponents(bool[] fill, int w, int h, int minArea)
    {
        int n = w * h;
        bool[] result  = new bool[n];
        bool[] visited = new bool[n];
        int[]  scratch = new int[n];
        var queue = new Queue<int>(256);

        for (int start = 0; start < n; start++)
        {
            if (!fill[start] || visited[start]) continue;

            int compSize = 0;
            int bx0 = w, by0 = h, bx1 = -1, by1 = -1;

            queue.Enqueue(start);
            visited[start] = true;

            while (queue.Count > 0)
            {
                int idx = queue.Dequeue();
                scratch[compSize++] = idx;

                int x = idx % w, y = idx / w;
                if (x < bx0) bx0 = x;
                if (y < by0) by0 = y;
                if (x > bx1) bx1 = x;
                if (y > by1) by1 = y;

                for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    if ((dx | dy) == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if ((uint)nx >= (uint)w || (uint)ny >= (uint)h) continue;
                    int nidx = ny * w + nx;
                    if (fill[nidx] && !visited[nidx])
                    {
                        visited[nidx] = true;
                        queue.Enqueue(nidx);
                    }
                }
            }

            // Drop if too few pixels (dust/specks)
            if (compSize < minArea) continue;

            // Drop if the bounding box is only 1 pixel wide or tall.
            // Shadow tendrils are often thin lines; real glyph strokes are always
            // at least 2 px wide in both dimensions at Blu-ray PGS render sizes.
            int bboxMinDim = Math.Min(bx1 - bx0 + 1, by1 - by0 + 1);
            if (bboxMinDim < 2) continue;

            for (int i = 0; i < compSize; i++)
                result[scratch[i]] = true;
        }
        return result;
    }

    // ── Gaussian blur (separable, float) ─────────────────────────────────────────────

    /// <summary>
    /// Separable Gaussian blur on a float[w*h] array.
    /// Kernel half-width = ceil(3*sigma); border pixels are clamped (replicated).
    /// </summary>
    static float[] GaussianBlur(float[] src, int w, int h, float sigma)
    {
        int half  = (int)MathF.Ceiling(3.0f * sigma);
        int kSize = 2 * half + 1;
        float[] k = new float[kSize];
        float   ksum = 0;
        for (int i = 0; i < kSize; i++)
        {
            float x = i - half;
            k[i] = MathF.Exp(-x * x / (2 * sigma * sigma));
            ksum += k[i];
        }
        for (int i = 0; i < kSize; i++) k[i] /= ksum;

        // Horizontal pass
        float[] tmp = new float[w * h];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float v = 0;
            for (int j = 0; j < kSize; j++)
            {
                int sx = Math.Clamp(x + j - half, 0, w - 1);
                v += k[j] * src[y * w + sx];
            }
            tmp[y * w + x] = v;
        }

        // Vertical pass
        float[] dst = new float[w * h];
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            float v = 0;
            for (int j = 0; j < kSize; j++)
            {
                int sy = Math.Clamp(y + j - half, 0, h - 1);
                v += k[j] * tmp[sy * w + x];
            }
            dst[y * w + x] = v;
        }
        return dst;
    }

    // ── morphological dilation (separable box) ────────────────────────────────────────

    static bool[] DilateBox(bool[] mask, int w, int h, int r)
    {
        if (r <= 0) return (bool[])mask.Clone();
        return DilateV(DilateH(mask, w, h, r), w, h, r);
    }

    static bool[] DilateH(bool[] src, int w, int h, int r)
    {
        bool[] dst = new bool[w * h];
        for (int y = 0; y < h; y++)
        {
            int off = y * w, count = 0;
            for (int k = 0; k <= Math.Min(r, w - 1); k++)
                if (src[off + k]) count++;
            dst[off] = count > 0;
            for (int x = 1; x < w; x++)
            {
                if (x + r     <  w && src[off + x + r    ]) count++;
                if (x - r - 1 >= 0 && src[off + x - r - 1]) count--;
                dst[off + x] = count > 0;
            }
        }
        return dst;
    }

    static bool[] DilateV(bool[] src, int w, int h, int r)
    {
        bool[] dst = new bool[w * h];
        for (int x = 0; x < w; x++)
        {
            int count = 0;
            for (int k = 0; k <= Math.Min(r, h - 1); k++)
                if (src[k * w + x]) count++;
            dst[x] = count > 0;
            for (int y = 1; y < h; y++)
            {
                if (y + r     <  h && src[(y + r    ) * w + x]) count++;
                if (y - r - 1 >= 0 && src[(y - r - 1) * w + x]) count--;
                dst[y * w + x] = count > 0;
            }
        }
        return dst;
    }

    // ── line splitting ────────────────────────────────────────────────────────────

    /// <summary>
    /// Splits a preprocessed Gray8 bitmap (dark text on white background) into individual
    /// line bitmaps by detecting horizontal runs of blank rows.
    ///
    /// A row is considered blank if no pixel is below <c>contentThreshold</c> (220).
    /// A gap of at least <paramref name="minGap"/> consecutive blank rows separates two lines.
    /// Lines shorter than <paramref name="minLineHeight"/> pixels are discarded as noise.
    ///
    /// Returns a single-element list containing the full image if no split point is found,
    /// so callers can always iterate the result uniformly.
    ///
    /// Each returned <c>SKBitmap</c> is caller-owned and must be disposed.
    /// </summary>
    public static List<(SKBitmap Bmp, int Y0, int Y1)> SplitLines(
        SKBitmap gray, int minGap = 2, int minLineHeight = 5)
    {
        int w = gray.Width, h = gray.Height;
        const byte contentThreshold = 220;

        // Build a per-row "has content" flag.
        var rowHasContent = new bool[h];
        unsafe
        {
            byte* px = (byte*)gray.GetPixels();
            for (int y = 0; y < h; y++)
            {
                int rowOff = y * w;
                for (int x = 0; x < w; x++)
                {
                    if (px[rowOff + x] < contentThreshold)
                    {
                        rowHasContent[y] = true;
                        break;
                    }
                }
            }
        }

        // Adaptive minGap: find the tallest contiguous content block first,
        // then require gaps to be at least 15 % of that height.  This prevents
        // intra-character gaps (e.g. the top stroke of え) from being treated
        // as line breaks while still splitting real multi-line subtitles.
        int maxContentRun = 0, curRun = 0;
        for (int y = 0; y < h; y++)
        {
            if (rowHasContent[y]) { curRun++; }
            else { if (curRun > maxContentRun) maxContentRun = curRun; curRun = 0; }
        }
        if (curRun > maxContentRun) maxContentRun = curRun;
        int adaptiveGap = Math.Max(minGap, (int)(maxContentRun * 0.15f));

        // Collect line regions: runs of content rows separated by ≥ adaptiveGap blank rows.
        var regions = new List<(int Start, int End)>();
        int startY = -1, gapCount = 0;

        for (int y = 0; y < h; y++)
        {
            if (rowHasContent[y])
            {
                if (startY == -1) startY = y;
                gapCount = 0;
            }
            else if (startY != -1)
            {
                gapCount++;
                if (gapCount >= adaptiveGap)
                {
                    int endY = y - gapCount + 1;
                    if (endY - startY >= minLineHeight)
                        regions.Add((startY, endY));
                    startY   = -1;
                    gapCount = 0;
                }
            }
        }
        if (startY != -1)
        {
            int endY = h;
            if (endY - startY >= minLineHeight)
                regions.Add((startY, endY));
        }

        // Fallback: return the whole image as a single line.
        if (regions.Count == 0)
            return [(gray.Copy(), 0, h)];

        // Crop each region into its own Gray8 bitmap.
        // Expand each content region by padPx on each side so that the returned
        // line images include white margin rather than starting exactly at the
        // first dark pixel.  The whitespace already present in the full image
        // (added by Preprocess) means the clamp never cuts the padding short.
        var lines = new List<(SKBitmap Bmp, int Y0, int Y1)>(regions.Count);
        unsafe
        {
            byte* src = (byte*)gray.GetPixels();
            foreach (var (start, end) in regions)
            {
                int contentH = end - start;
                int padPx    = Math.Max(14, (int)MathF.Round(PadFraction * contentH));
                int cropY0   = Math.Max(0, start - padPx);
                int cropY1   = Math.Min(h, end   + padPx);
                int lineH    = cropY1 - cropY0;

                // Horizontal tight-crop: find left/right content bounds in this line's rows.
                int colX0 = w, colX1 = -1;
                for (int y = start; y < end; y++)
                {
                    for (int x = 0; x < w; x++)
                        if (src[y * w + x] < contentThreshold) { if (x < colX0) colX0 = x; break; }
                    for (int x = w - 1; x >= 0; x--)
                        if (src[y * w + x] < contentThreshold) { if (x > colX1) colX1 = x; break; }
                }
                int cropX0 = colX1 < 0 ? 0 : Math.Max(0, colX0 - padPx);
                int cropX1 = colX1 < 0 ? w : Math.Min(w, colX1 + 1 + padPx);
                int lineW  = cropX1 - cropX0;

                var line  = new SKBitmap(lineW, lineH, SKColorType.Gray8, SKAlphaType.Opaque);
                byte* dst = (byte*)line.GetPixels();

                // Copy the full crop region from the source image.
                for (int row = 0; row < lineH; row++)
                    Buffer.MemoryCopy(src + (cropY0 + row) * w + cropX0, dst + row * lineW, lineW, lineW);

                // Blank outer margin rows so closely-spaced neighbouring lines
                // don't bleed into this line's padding.  Keep a small buffer
                // around the content edges to preserve descenders / ascenders.
                int guardPx  = Math.Max(2, contentH / 8);
                int blankTop = start - guardPx - cropY0;   // row index within line bitmap
                int blankBot = end   + guardPx - cropY0;
                for (int row = 0; row < Math.Min(blankTop, lineH); row++)
                    new Span<byte>(dst + row * lineW, lineW).Fill(255);
                for (int row = Math.Max(blankBot, 0); row < lineH; row++)
                    new Span<byte>(dst + row * lineW, lineW).Fill(255);

                lines.Add((line, cropY0, cropY1));
            }
        }
        return lines;
    }

    // ── scale / pixel helpers ─────────────────────────────────────────────────────────

    static byte[] ScaleGray(byte[] src, int srcW, int srcH, int dstW, int dstH)
    {
        using var srcBmp = MakeGray8(src, srcW, srcH);
        using var dstBmp = srcBmp.Resize(
            new SKImageInfo(dstW, dstH, SKColorType.Gray8, SKAlphaType.Opaque),
            new SKSamplingOptions(SKCubicResampler.Mitchell));

        if (dstBmp == null)
        {
            var blank = new byte[dstW * dstH];
            Array.Fill(blank, (byte)255);
            return blank;
        }

        byte[] dst = new byte[dstW * dstH];
        unsafe
        {
            byte* ptr = (byte*)dstBmp.GetPixels();
            for (int i = 0; i < dst.Length; i++) dst[i] = ptr[i];
        }
        return dst;
    }

    static SKBitmap MakeGray8(byte[] pixels, int w, int h)
    {
        var bmp = new SKBitmap(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
        unsafe
        {
            byte* ptr = (byte*)bmp.GetPixels();
            for (int i = 0; i < pixels.Length; i++) ptr[i] = pixels[i];
        }
        return bmp;
    }

    static int BucketHeight(int h) => h switch
    {
        < 90  => BucketSmall,
        < 140 => BucketMedium,
        _     => BucketLarge
    };

    static byte Luma(byte r, byte g, byte b) =>
        (byte)(0.299f * r + 0.587f * g + 0.114f * b);

    // ── crop helpers ──────────────────────────────────────────────────────────────────

    static bool TryBBox(bool[] mask, int w, int h,
        out int x0, out int y0, out int x1, out int y1)
    {
        x0 = w; y0 = h; x1 = -1; y1 = -1;
        for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            if (!mask[y * w + x]) continue;
            if (x < x0) x0 = x;
            if (y < y0) y0 = y;
            if (x > x1) x1 = x;
            if (y > y1) y1 = y;
        }
        return x1 >= 0;
    }

    static bool[] CropBool(bool[] src, int srcW, int x0, int y0, int cw, int ch)
    {
        bool[] dst = new bool[cw * ch];
        for (int row = 0; row < ch; row++)
            Array.Copy(src, (y0 + row) * srcW + x0, dst, row * cw, cw);
        return dst;
    }

    static byte[] CropByte(byte[] src, int srcW, int x0, int y0, int cw, int ch)
    {
        byte[] dst = new byte[cw * ch];
        for (int row = 0; row < ch; row++)
            Array.Copy(src, (y0 + row) * srcW + x0, dst, row * cw, cw);
        return dst;
    }

    // ── debug save helpers ────────────────────────────────────────────────────────────

    static void SaveDebugRgba(SKBitmap bmp, Options opts, string stage)
    {
        if (opts.DebugDir == null) return;
        SavePng(bmp, opts, stage);
    }

    static void SaveDebugRgbaCrop(SKBitmap src, int x0, int y0, int cw, int ch,
        Options opts, string stage)
    {
        if (opts.DebugDir == null) return;
        using var cropped = new SKBitmap(cw, ch, src.ColorType, src.AlphaType);
        unsafe
        {
            byte* sp = (byte*)src.GetPixels();
            byte* dp = (byte*)cropped.GetPixels();
            int ss = src.Width * 4, ds = cw * 4;
            for (int row = 0; row < ch; row++)
                Buffer.MemoryCopy(sp + (y0 + row) * ss + x0 * 4, dp + row * ds, ds, ds);
        }
        SavePng(cropped, opts, stage);
    }

    static void SaveDebugGray(byte[] gray, int w, int h, Options opts, string stage)
    {
        if (opts.DebugDir == null) return;
        using var bmp = MakeGray8(gray, w, h);
        SavePng(bmp, opts, stage);
    }

    static void SaveDebugMask(bool[] mask, int w, int h, Options opts, string stage)
    {
        if (opts.DebugDir == null) return;
        byte[] gray = new byte[w * h];
        for (int i = 0; i < w * h; i++)
            gray[i] = mask[i] ? (byte)0 : (byte)255;
        SaveDebugGray(gray, w, h, opts, stage);
    }

    static void SavePng(SKBitmap bmp, Options opts, string stage)
    {
        string prefix = opts.DebugPrefix != null ? $"{opts.DebugPrefix}_" : "";
        string path   = Path.Combine(opts.DebugDir!, $"{prefix}{stage}.png");
        using var data   = bmp.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.OpenWrite(path);
        data.SaveTo(stream);
    }
}
