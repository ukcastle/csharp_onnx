using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
namespace onnx_test
{
    internal class OnnxMMpose : OnnxModel
    {

        /*
         * 100회 기준 tact time
         * hr : 97ms, 10fps / 33ms, 30fps
         * res50fp16 : 65ms, 15fps / 26ms, 38fps
         * mobv2 : 48ms, 20fps /  14ms, 71fps
         * shufflev2 : 33ms, 30fps / 16ms, 62fps
         */

        // Member
        private static readonly int[,] skeleton = new int[16, 5]
        {
            { 10, 8, 0, 255, 0 }, 
            { 8, 6, 0, 255, 0 }, 
            { 11, 9, 255, 128, 0 },
            { 9, 7, 255, 128, 0 },
            { 6, 7, 51, 153, 255 }, 
            { 0, 6, 51, 153, 255 },
            { 1, 7, 51, 153, 255 }, 
            { 0, 1, 51, 153, 255 },
            { 0, 2, 0, 255, 0 },
            { 1, 3, 255, 128, 0 },
            { 2, 4, 0, 255, 0 },
            { 3, 5, 255, 128, 0 }, 
            { 5, 12, 51, 153, 153 },
            { 4, 12, 51, 153, 153 }, 
            { 12, 13, 0, 0, 255 },
            { 13, 14, 0, 255, 255 }
        };
        private enum KeyPoint
        {
            LeftShoulder=0,
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
            RightAnkle,
            Grip,
            Shaft,
            ClubHead
        }

        // Constructor
        public OnnxMMpose(string onnxPath, int width = 192, int height = 256, string inputName = "input.1")
            : base(onnxPath, width, height, inputName) { }

        public override List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, int imgWidth, int imgHeight, int bboxX = 0, int bboxY = 0, int batchIdx = 0)
        {
            var predValue = output[0].AsEnumerable<float>().ToArray(); // 1 17 64 48
            var predDims = output[0].AsTensor<float>().Dimensions.ToArray(); // 1 17 64 48

            int scaleX = imgWidth / predDims[3];
            int scaleY = imgHeight / predDims[2];

            var keyPoints = new List<List<float>>(); // 17(key) * 3(x,y,pred)

            for (int key = 0; key < predDims[1]; key++)
            {
                int idx1 = batchIdx * predDims[1] * predDims[2] * predDims[3];
                int idx2 = key * predDims[2] * predDims[3];
                var keyValue = OnnxMMpose.Heatmap2KeyPoint(predValue, idx1 + idx2, predDims[2], predDims[3]); // 3(x,y,pred);
                OnnxModel.RefinePointBySize(ref keyValue, scaleX, scaleY, bboxX, bboxY);
                keyPoints.Add(keyValue);
            }

            return this.FitSizeofOutput(ref keyPoints, ref ratio, ref diff);
        }

        public override Mat DrawOutput(ref Mat inputMat, List<List<float>> output)
        {
            Mat outputMat = inputMat.Clone();

            for (int i = 0; i < (skeleton.Length / 5); i++)
            {
                int startIdx = OnnxMMpose.skeleton[i, 0];
                int endIdx = OnnxMMpose.skeleton[i, 1];
                Scalar color = new Scalar(skeleton[i, 2], skeleton[i, 3], skeleton[i, 4]);
                Point start = new Point(output[startIdx][0], output[startIdx][1]);
                Point end = new Point(output[endIdx][0], output[endIdx][1]);

                Cv2.Line(outputMat, start, end, color);
            }

            return outputMat;
        }

        public void FitOriginSize(ref List<List<float>> output, int baseX1, int baseY1)
        {
            for (int i=0; i < output.Count; i++)
            {
                output[i][0] += baseX1;
                output[i][1] += baseY1;
            }
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

            keyValue.Add(OnnxMMpose.RefinePoint(heatmap, topX, absIdx, heatmapHeight, heatmapWidth, isY: false));  // normalized x
            keyValue.Add(OnnxMMpose.RefinePoint(heatmap, topY, absIdx, heatmapHeight, heatmapWidth, isY: true));   // normalized y
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

        protected override List<List<float>> FitSizeofOutput(ref List<List<float>> output, ref float ratio, ref Point diff)
        {
            var outputList = new List<List<float>>();

            for (int i = 0; i < output.Count; i++)
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
