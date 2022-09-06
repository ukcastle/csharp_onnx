using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace onnx_test
{
    class Onnx_MMpose
    {
        // Member
        private static readonly int[,] skeleton = new int[17, 5]
        {
            { 15, 13, 255, 128, 0 }, // Left Ankle - Left Knee
            { 16, 14, 255, 128, 0 }, // Right Ankle - Right Knee
            { 11, 13, 255, 0, 0 }, // Left Knee - Left Hip
            { 12, 14, 255, 0, 0 }, // Right Knee - Right Hip
            { 11, 12, 255, 255, 0 }, // Left Hip - Right Hip
            { 5, 11, 0, 255, 0 }, // Left Shoulder - Left Hip
            { 6, 12, 0, 255, 0 }, // Right Shoulder - Right Hip
            { 5, 7, 0, 255, 255 }, // Left Shoulder - Left Elbow
            { 6, 8, 0, 255, 255 }, // Right Shoulder - Right Elbow
            { 9, 7, 255, 0, 255 }, // Left Wrist - Left Elbow
            { 10, 8, 255, 0, 255 }, // Right Wrist - Right Elbow
            { 5, 1, 128, 128, 255 }, // Left Shoulder - Head Bottom
            { 6, 1, 128, 128, 255 }, // Right Shoulder - Head Bottom
            { 0, 1, 128, 128, 128 }, // Nose - Head Bottom
            { 0, 2, 128, 128, 128 }, // Nose - Head Top
            { 0, 3, 0, 0, 128 }, // Nose - Left Ear
            { 0, 4, 0, 0, 128 } // Nose - Right Ear
        };
        private enum KeyPoint
        {
            Nose = 0,
            HeadBottom,
            HeadTop,
            LeftEar,
            RightEar,
            LeftShoulder,
            RightShoulder,
            LeftElbow,
            RightElbow,
            LeftWrist,
            RightWrist,
            LeftHip,
            RightHip,
            LeftKnee,
            RightKnee,
            LeftAnkle,
            RightAnkle
        }
        private readonly InferenceSession sess;
        private readonly Size size;
        private readonly Scalar backgroundColor;
        private readonly int keyPointLength;
        private static readonly float[,] normalizeValue = new float[2, 3]
        {
            {0.406f, 0.456f, 0.485f}, // mean b g r
            {0.225f, 0.224f, 0.229f} // std b g r
        };

        // Constructor
        public Onnx_MMpose(string onnxPath, int width, int height, int backgroundColor = 114)
        {
            if (string.IsNullOrWhiteSpace(onnxPath))
            {
                throw new ArgumentException("message", nameof(onnxPath));
            }

            var option = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,    
            };

            //SessionOptions.MakeSessionOptionWithCudaProvider();
            
            this.sess = new InferenceSession(onnxPath, option);
            //this.sess = new InferenceSession(onnxPath, SessionOptions.MakeSessionOptionWithCudaProvider());
            this.size = new Size(width, height);
            this.backgroundColor = new Scalar(backgroundColor, backgroundColor, backgroundColor);
            this.keyPointLength = System.Enum.GetValues(typeof(KeyPoint)).Length;
        }

        // Public APIs
        public Mat MakeInputMat(ref Mat input, out float ratio, out Point diff, out Point diff2, bool auto = true, bool scaleFill = false, bool isScaleUp = true)
        {
            Mat img = input.Clone();
            //Cv2.CvtColor(img, img, ColorConversionCodes.BGR2RGB);
            ratio = Math.Min((float)this.size.Width / img.Width, (float)this.size.Height / img.Height);

            if (!isScaleUp)
            {
                ratio = Math.Min(ratio, 1.0f);
            }

            var unpad = new Size((int)Math.Round(img.Width * ratio), (int)Math.Round(img.Height * ratio));

            var dW = this.size.Width - unpad.Width;
            var dH = this.size.Height - unpad.Height;

            var tensorRatio = this.size.Height / (float)this.size.Width;
            var inputRatio = input.Height / (float)input.Width;

            if (auto && tensorRatio != inputRatio)
            {
                dW %= 32;
                dH %= 32;
            }
            else if (scaleFill)
            {
                dW = 0;
                dH = 0;
                unpad = this.size;
            }

            var dW_h = (int)Math.Round((float)dW / 2);
            var dH_h = (int)Math.Round((float)dH / 2);
            var dw2 = 0;
            var dh2 = 0;
            if (dW_h * 2 != dW)
            {
                dw2 = dW - dW_h * 2;
            }
            if (dH_h * 2 != dH)
            {
                dh2 = dH - dH_h * 2;
            }

            if (img.Width != unpad.Width || img.Height != unpad.Height)
            {
                Cv2.Resize(img, img, unpad);
            }
            Cv2.CopyMakeBorder(img, img, dH_h + dh2, dH_h,
                               dW_h + dw2, dW_h, BorderTypes.Constant, this.backgroundColor);
            diff = new Point(dW_h, dH_h);
            diff2 = new Point(dw2, dh2);
            return img;
        }

        public Mat MakeInputMat(string inputPath, out Mat srcMat, out float ratio, out Point diff, out Point diff2)
        {
            srcMat = Cv2.ImRead(inputPath, ImreadModes.Color);
            return this.MakeInputMat(ref srcMat, out ratio, out diff, out diff2, auto: false, scaleFill: false);
        }

        public DisposableNamedOnnxValue[] ModelRun(ref Mat inputMat)
        {
            Mat mat = new Mat();
            inputMat.ConvertTo(mat, MatType.CV_32FC3, (float)(1 / 255.0));
            var onnxInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input.1", new DenseTensor<float>(Onnx_MMpose.Mat2Array(mat), new[] { 1, 3, mat.Height, mat.Width }))
            };
            return this.sess.Run(onnxInput).ToArray();
        }

        public Mat DrawOutput(ref Mat inputMat, List<List<float>> output)
        {
            Mat outputMat = inputMat.Clone();

            for (int i = 0; i < this.keyPointLength; i++)
            {
                int startIdx = Onnx_MMpose.skeleton[i, 0];
                int endIdx = Onnx_MMpose.skeleton[i, 1];
                Scalar color = new Scalar(skeleton[i, 2], skeleton[i, 3], skeleton[i, 4]);

                Point start = new Point(output[startIdx][0], output[startIdx][1]);
                Point end = new Point(output[endIdx][0], output[endIdx][1]);

                Cv2.Line(outputMat, start, end, color);
            }

            return outputMat;
        }

        public List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, ref Point diff2,  int imgWidth, int imgHegiht, int bboxX = 0, int bboxY = 0, int batchIdx = 0)
        {
            var predValue = output[0].AsEnumerable<float>().ToArray(); // 1 17 64 48
            var predDims = output[0].AsTensor<float>().Dimensions.ToArray(); // 1 17 64 48

            var keyPointsBatch = new List<List<List<float>>>(); // batch * 17(key) * 3(x,y,pred)

            int scaleX = imgWidth / predDims[3];
            int scaleY = imgHegiht / predDims[2];

            for (int batch = 0; batch < predDims[0]; batch++)
            {
                var keyPoints = new List<List<float>>(); // 17(key) * 3(x,y,pred)

                for (int key = 0; key < predDims[1]; key++)
                {
                    int idx1 = batch * predDims[1] * predDims[2] * predDims[3]; //사실 0만나옴
                    int idx2 = key * predDims[2] * predDims[3];
                    var keyValue = Onnx_MMpose.Heatmap2KeyPoint(predValue, idx1 + idx2, predDims[2], predDims[3]); // 3(x,y,pred);
                    Onnx_MMpose.RefinePointBySize(ref keyValue, scaleX, scaleY, bboxX, bboxY);
                    keyPoints.Add(keyValue);
                }
                keyPointsBatch.Add(keyPoints);
            }
            return this.FitSizeofOutput(ref keyPointsBatch, ref ratio, ref diff, ref diff2);
        }

        // Private Func
        private static void RefinePointBySize(ref List<float> keyValue, int scaleX, int scaleY, int bboxX, int bboxY)
        {
            keyValue[0] = keyValue[0] * scaleX + bboxX; // x
            keyValue[1] = keyValue[1] * scaleY + bboxY; // y
        }

        private static List<float> Heatmap2KeyPoint(float[] heatmap, int keyIdx, int heatmapHeight, int heatmapWidth)
        {
            var keyValue = new List<float>();
            int topX = 0, topY = 0, absIdx = 0;
            float topValue = 0.0f;
            for (int i = keyIdx; i < keyIdx + heatmapHeight * heatmapWidth; i++) // argmax, amax 로 좌표 저장
            {
                if (heatmap[i] > topValue)
                {
                    topValue = heatmap[i];
                    absIdx = i;
                    topX = (i - keyIdx) % heatmapWidth;
                    topY = (i - keyIdx) / heatmapWidth;
                }
            }

            keyValue.Add(Onnx_MMpose.RefinePoint(heatmap, topX, absIdx, heatmapHeight, heatmapWidth, isY: false));  // normalized x
            keyValue.Add(Onnx_MMpose.RefinePoint(heatmap, topY, absIdx, heatmapHeight, heatmapWidth, isY: true));   // normalized y
            keyValue.Add(topValue); // pred

            return keyValue;
        }

        private static float RefinePoint(float[] heatmap, int point, int absIdx, int heatmapHeight, int heatmapWidth, bool isY = false)
        {
            if ((point < 1) || (!isY && point >= heatmapWidth - 1) || (isY && point >= heatmapHeight - 1))
            {
                return point;
            }

            int step = isY ? heatmapWidth : 1;

            float tempPoint = point;

            if ((heatmap[absIdx + step] - heatmap[absIdx - step]) >= 0)
            {
                tempPoint += 0.25f;
            }
            else
            {
                tempPoint -= 0.25f;
            }
            return tempPoint;
        }

        private static unsafe float[] Mat2Array(Mat mat)
        {
            var imgHeight = mat.Height;
            var imgWidth = mat.Width;
            var imgChannel = mat.Channels();

            float* matPointer = (float*)mat.DataPointer;

            float[] array = new float[imgHeight * imgWidth * imgChannel]; // H * W * C

            for (int y = 0; y < imgHeight; y++)
            {
                for (int x = 0; x < imgWidth; x++)
                {
                    for (int c = 0; c < imgChannel; c++)
                    {
                        var baseIdx = (y * imgChannel) * imgWidth + (x * imgChannel) + imgChannel;
                        var convertedIdx = (c * imgWidth) * imgHeight + (y * imgWidth) + x;
                        array[convertedIdx] = matPointer[baseIdx];
                        array[convertedIdx] = (matPointer[baseIdx] - Onnx_MMpose.normalizeValue[0, c]) / Onnx_MMpose.normalizeValue[1, c];
                    }
                }
            }
            return array;
        }

        private List<List<float>> FitSizeofOutput(ref List<List<List<float>>> output, ref float ratio, ref Point diff, ref Point diff2, int batchIdx = 0)
        {
            var outputList = new List<List<float>>();

            for (int i=0; i< this.keyPointLength; i++)
            {
                var curOutput = output[batchIdx][i];
                int fitX = (int)(Math.Max(curOutput[0] - diff.X, 0) / ratio);
                int fitY = (int)(Math.Max(curOutput[1] - diff.Y, 0) / ratio);

                outputList.Add(new List<float> { fitX, fitY, curOutput[2] });
            }

            return outputList;
        }

    }
}
