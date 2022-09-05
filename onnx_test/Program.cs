using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


namespace onnx_test
{
    static class Program
    {
        
        [STAThread]
        static void Main()
        {
            int[,] skeleton = new int[17, 5]
            {
                {15, 13, 255, 128, 0 },
                {16, 14, 255, 128, 0 },
                {12, 14, 255, 0, 0 },
                {11, 13, 255, 0, 0 },
                {11, 12, 255, 255, 0 },
                {6, 12, 0, 255, 0 },
                {5, 11, 0, 255, 0 },
                {6, 8, 0, 255, 255 },
                {5, 7, 0, 255, 255 },
                {10, 8, 255, 0, 255 },
                {9, 7, 255, 0, 255 },
                {6, 1, 128, 128, 255 },
                {5, 1, 128, 128, 255 },
                {0, 1, 128, 128, 128 },
                {0, 2, 128, 128, 128 },
                {0, 3, 0, 0, 128 },
                {0, 4, 0, 0, 128 }
            };
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            //Application.Run(new Form1());
            Onnx_MMpose onnxPose = new Onnx_MMpose("C:\\Users\\Admin\\Documents\\output.onnx", 192, 256);

            Mat src = Cv2.ImRead("C:\\Users\\Admin\\Documents\\3326.jpg", ImreadModes.Color);
            var inputMat = onnxPose.MakeLetterBoxByMat(ref src, out float ratio, out Point diff, out Point diff2, auto:false, scaleFill:false);
            var results = onnxPose.ModelRun(ref inputMat); // 1(N) * 17(C) * 64(H) * 48(W)
            

            var predValue = results[0].AsEnumerable<float>().ToArray(); // 1 17 64 48
            var predDims = results[0].AsTensor<float>().Dimensions.ToArray(); // 1 17 64 48

            var output = PostProcessing(predValue, predDims, inputMat.Width, inputMat.Height);

            for (int i=0; i<17; i++)
            {
                int startIdx = skeleton[i,0];
                int endIdx = skeleton[i,1];
                Scalar color = new Scalar(skeleton[i, 2], skeleton[i, 3], skeleton[i, 4]);

                Point start = new Point(output[0][startIdx][0], output[0][startIdx][1]);
                Point end = new Point(output[0][endIdx][0], output[0][endIdx][1]);

                Cv2.Line(inputMat, start, end, color);
            }
            Cv2.Line(inputMat, diff, diff2, new Scalar(255,0,0));
            Cv2.Resize(inputMat, inputMat, new Size(192 * 3, 256 * 3));
            Cv2.ImShow("dd", inputMat);
            Cv2.WaitKey();
        }

        static List<List<List<float>>> PostProcessing(float[] predVlaue, int[] predDims, int imgWidth, int imgHegiht, int bboxX = 0, int bboxY = 0)
        {
            var keyPointsBatch = new List<List<List<float>>>(); // batch * 17(key) * 3(x,y,pred)

            int scaleX = imgWidth / predDims[3];
            int scaleY = imgHegiht / predDims[2];

            for (int batch=0; batch < predDims[0]; batch++)
            {
                var keyPoints = new List<List<float>>(); // 17(key) * 3(x,y,pred)

                for (int key=0; key<predDims[1]; key++)
                {
                    int idx1 = batch * predDims[1] * predDims[2] * predDims[3]; //사실 0만나옴
                    int idx2 = key * predDims[2] * predDims[3];
                    var keyValue = Heatmap2KeyPoint(predVlaue, idx1+idx2, predDims[2], predDims[3]); // 3(x,y,pred);
                    refinePointBySize(ref keyValue, scaleX, scaleY, bboxX, bboxY);
                    keyPoints.Add(keyValue);
                }


                keyPointsBatch.Add(keyPoints);
            }

            return keyPointsBatch;
        }

        static void refinePointBySize(ref List<float> keyValue, int scaleX, int scaleY, int bboxX, int bboxY)
        {
            keyValue[0] = keyValue[0] * scaleX + bboxX; // x
            keyValue[1] = keyValue[1] * scaleY + bboxY; // y
        }

        static List<float> Heatmap2KeyPoint(float[] heatmap, int keyIdx, int heatmapHeight, int heatmapWidth)
        {
            var keyValue = new List<float>();
            int topX = 0, topY = 0, absIdx = 0;
            float topValue = 0.0f ;
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

            keyValue.Add(RefinePoint(heatmap, topX, absIdx, heatmapHeight, heatmapWidth, isY: false));  // normalized x
            keyValue.Add(RefinePoint(heatmap, topY, absIdx, heatmapHeight, heatmapWidth, isY: true));   // normalized y
            keyValue.Add(topValue); // pred

            return keyValue;
        }

        static float RefinePoint(float[] heatmap ,int point, int absIdx, int heatmapHeight, int heatmapWidth,  bool isY= false)
        {
            if ( (point < 1) || (!isY && point >= heatmapWidth - 1) || (isY && point >= heatmapHeight -1))
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
        
    }
}