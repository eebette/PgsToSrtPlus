using Nikse.SubtitleEdit.Core.BluRaySup;
using SkiaSharp;

namespace PgsToSrtPlus;

/// <summary>
/// Decodes PGS bitmap data from libse's parsed structures using SkiaSharp.
/// Replaces System.Drawing-based GetBitmap() which doesn't work on Linux.
/// </summary>
static class PgsDecoder
{
    /// <summary>
    /// Decodes a PCS display set into an RGBA SKBitmap.
    /// Returns null if the display set contains no renderable content.
    /// </summary>
    public static SKBitmap? DecodePgsImage(BluRaySupParser.PcsData pcsData)
    {
        if (pcsData.PcsObjects.Count == 0 || pcsData.BitmapObjects.Count == 0)
            return null;

        var palette = BluRaySupParser.SupDecoder.DecodePalette(pcsData.PaletteInfos);

        var pcsObj = pcsData.PcsObjects[0];
        if (pcsObj.ObjectId >= pcsData.BitmapObjects.Count)
            return null;
        return DecodeObject(pcsData.BitmapObjects[pcsObj.ObjectId], palette);
    }

    static SKBitmap? DecodeObject(List<BluRaySupParser.OdsData> data, BluRaySupPalette palette)
    {
        if (data.Count == 0)
            return null;

        var ods = data[0];
        int w = ods.Size.Width, h = ods.Size.Height;
        if (w == 0 || h == 0)
            return null;

        byte[] rleData = ods.Fragment.ImageBuffer;
        var bmp = new SKBitmap(w, h, SKColorType.Bgra8888, SKAlphaType.Premul);
        var pixels = bmp.GetPixels();

        int x = 0, y = 0;
        int idx = 0;

        unsafe
        {
            byte* ptr = (byte*)pixels;

            while (idx < rleData.Length && y < h)
            {
                byte b = rleData[idx++];
                if (b != 0)
                {
                    // Single pixel of palette index b
                    SetPixel(ptr, w, x, y, palette.GetArgb(b));
                    x++;
                }
                else
                {
                    if (idx >= rleData.Length) break;
                    byte flags = rleData[idx++];
                    if (flags == 0)
                    {
                        // End of line
                        x = 0;
                        y++;
                    }
                    else
                    {
                        int runLength;
                        int colorIndex;

                        if ((flags & 0xC0) == 0)
                        {
                            // 00 01-3F: short run of color 0
                            runLength = flags;
                            colorIndex = 0;
                        }
                        else if ((flags & 0xC0) == 0x40)
                        {
                            // 00 40-7F NN: long run of color 0
                            if (idx >= rleData.Length) break;
                            runLength = ((flags & 0x3F) << 8) | rleData[idx++];
                            colorIndex = 0;
                        }
                        else if ((flags & 0xC0) == 0x80)
                        {
                            // 00 80-BF CC: short run of color CC
                            if (idx >= rleData.Length) break;
                            runLength = flags & 0x3F;
                            colorIndex = rleData[idx++];
                        }
                        else
                        {
                            // 00 C0-FF NN CC: long run of color CC
                            if (idx + 1 >= rleData.Length) break;
                            runLength = ((flags & 0x3F) << 8) | rleData[idx++];
                            colorIndex = rleData[idx++];
                        }

                        int argb = palette.GetArgb(colorIndex);
                        for (int i = 0; i < runLength && x < w; i++, x++)
                            SetPixel(ptr, w, x, y, argb);
                    }
                }

                // PGS RLE uses explicit 00 00 end-of-line markers; do not auto-wrap.
            }
        }

        return bmp;
    }

    static unsafe void SetPixel(byte* ptr, int stride, int x, int y, int argb)
    {
        int offset = (y * stride + x) * 4;
        ptr[offset]     = (byte)argb;        // B
        ptr[offset + 1] = (byte)(argb >> 8);   // G
        ptr[offset + 2] = (byte)(argb >> 16);  // R
        ptr[offset + 3] = (byte)(argb >> 24);  // A
    }
}
