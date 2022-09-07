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
            //{0.485f, 0.456f, 0.406f}, // mean r g b
            //{0.229f, 0.224f, 0.225f} // std r g b
        };

        // Constructor
        public Onnx_MMpose(string onnxPath, int width, int height, int backgroundColor = 114)
        {
            var option = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                //GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC, // 조금 더 걸림
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };

            this.sess = new InferenceSession(onnxPath, option);
            this.size = new Size(width, height);
            this.backgroundColor = new Scalar(backgroundColor, backgroundColor, backgroundColor);
            this.keyPointLength = System.Enum.GetValues(typeof(KeyPoint)).Length;
        }

        // Public APIs
        public Mat MakeInputMat(ref Mat input, out float ratio, out Point diff, bool isScaleUp = true)
        {
            /*
             * input : Mat ( Do not CvtColor )
             * Output : Mat ( Add padding ), 
             * Out Ref : ratio, diff => Used PostProcessing For fit x,y 
             * isScaleUp => If Input Image is less than Model Input, Upscale Input Image
             */

            Mat img = input.Clone();
            //Cv2.CvtColor(img, img, ColorConversionCodes.BGR2RGB);
            ratio = Math.Min((float)this.size.Width / img.Width, (float)this.size.Height / img.Height);

            if (!isScaleUp)
            {
                ratio = Math.Min(ratio, 1.0f);
            }

            var unpadImageSize = new Size((int)Math.Round(img.Width * ratio), (int)Math.Round(img.Height * ratio));

            var backgroundWidth = this.size.Width - unpadImageSize.Width;
            var backgroundHeigth = this.size.Height - unpadImageSize.Height;

            var halfBackgroundWidth = (int)Math.Round((float)backgroundWidth / 2);
            var halfBackgroundHeigth = (int)Math.Round((float)backgroundHeigth / 2);
            var ifBackgroundWidthIsOdd = 0;
            var ifBackgroundHeightIsOdd = 0;
            if (halfBackgroundWidth * 2 != backgroundWidth)
            {
                ifBackgroundWidthIsOdd = backgroundWidth - halfBackgroundWidth * 2;
            }
            if (halfBackgroundHeigth * 2 != backgroundHeigth)
            {
                ifBackgroundHeightIsOdd = backgroundHeigth - halfBackgroundHeigth * 2;
            }

            if (img.Width != unpadImageSize.Width || img.Height != unpadImageSize.Height)
            {
                Cv2.Resize(img, img, unpadImageSize);
            }
            Cv2.CopyMakeBorder(img, img, halfBackgroundHeigth + ifBackgroundHeightIsOdd, halfBackgroundHeigth,
                               halfBackgroundWidth + ifBackgroundWidthIsOdd, halfBackgroundWidth, BorderTypes.Constant, this.backgroundColor);
            diff = new Point(halfBackgroundWidth + ifBackgroundWidthIsOdd, halfBackgroundHeigth+ifBackgroundHeightIsOdd);
            return img;
        }

        public Mat MakeInputMat(string inputPath, out Mat srcMat, out float ratio, out Point diff)
        {
            srcMat = Cv2.ImRead(inputPath, ImreadModes.Color);
            return this.MakeInputMat(ref srcMat, out ratio, out diff);
        }

        public DisposableNamedOnnxValue[] ModelRun(ref Mat inputMat)
        {
            /*
             * input : Mat ( Have to fit Model Input )
             * Output : DisposableNamedOnnxValue[] ( 1(N) * 17(C) * 64(H) * 48(W) )
             */

            Mat mat = new Mat();
            inputMat.ConvertTo(mat, MatType.CV_32FC3, (float)(1 / 255.0));
            var onnxInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input.1", new DenseTensor<float>(Onnx_MMpose.Mat2Array(mat), new[] { 1, 3, mat.Height, mat.Width }))
            };
            return this.sess.Run(onnxInput).ToArray();
        }

        public List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, int imgWidth, int imgHeight, int bboxX = 0, int bboxY = 0, int batchIdx = 0)
        {
            var predValue = output[0].AsEnumerable<float>().ToArray(); // 1 17 64 48
            var predDims = output[0].AsTensor<float>().Dimensions.ToArray(); // 1 17 64 48

            int scaleX = imgWidth / predDims[3];
            int scaleY = imgHeight / predDims[2];

            var keyPoints = new List<List<float>>(); // 17(key) * 3(x,y,pred)

            for (int key = 0; key < predDims[1]; key++)
            {
                int idx1 = batchIdx * predDims[1] * predDims[2] * predDims[3]; //사실 0만나옴
                int idx2 = key * predDims[2] * predDims[3];
                var keyValue = Onnx_MMpose.Heatmap2KeyPoint(predValue, idx1 + idx2, predDims[2], predDims[3]); // 3(x,y,pred);
                Onnx_MMpose.RefinePointBySize(ref keyValue, scaleX, scaleY, bboxX, bboxY);
                keyPoints.Add(keyValue);
            }

            return this.FitSizeofOutput(ref keyPoints, ref ratio, ref diff);
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
            /*
             * 기존엔 가우시안 분포를 이용하여 값을 추출함
             * 연산시간 많이들어서 좀 더 짧은 후처리 방법 사용
             */
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

        private List<List<float>> FitSizeofOutput(ref List<List<float>> output, ref float ratio, ref Point diff)
        {
            var outputList = new List<List<float>>();

            for (int i=0; i< this.keyPointLength; i++)
            {
                var curOutput = output[i];
                int fitX = (int)(Math.Max(curOutput[0] - diff.X, 0) / ratio);
                int fitY = (int)(Math.Max(curOutput[1] - diff.Y, 0) / ratio);

                outputList.Add(new List<float> { fitX, fitY, curOutput[2] });
            }

            return outputList;
        }

    }
}
