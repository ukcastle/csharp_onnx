using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;

using System.Diagnostics;

namespace onnx_test
{
    static class Program
    {

        [STAThread]
        static void Main()
       {
            const string posePath = "C:\\Users\\Admin\\Documents\\hr.onnx";
            const string yoloPath = "C:\\Users\\Admin\\Documents\\640s_full.onnx";
            const string imgPath = "C:\\Users\\Admin\\Documents\\sdsdsd.jpg";

            /* 모델 로드 */
            OnnxMMpose onnxModel = new OnnxMMpose(posePath, 192, 256);
            OnnxYolo yoloModel = new OnnxYolo(yoloPath, 640, 640);

            Mat src = Cv2.ImRead(imgPath); 

            /* 전체 이미지에서 사람 부분만 크롭해오기 */
            /* baseX1,Y1에 크롭된 왼쪽 위 꼭짓점 좌표 받아옴(원본 영상에 매칭하기 위해) */
            var croppedImg = RunYoloDetect(ref yoloModel, ref src, out int baseX1, out int baseY1);
            
            /* PostProcess까지 하면 결과값 출력 */
            /* KeyPoint 순서로 출력됨(OnnxMMpose.cs 참조) */
            var inputMat = onnxModel.MakeInputMat(ref croppedImg, out float ratio, out Point diff);
            var results = onnxModel.ModelRun(ref inputMat);
            var output = onnxModel.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height);

            /* Crop된 이미지 기준으로 Inference한 좌표를 원본 좌표에 맞추기 */
            onnxModel.FitOriginSize(ref output, baseX1, baseY1);

            /* 추가옵션(시각화) */
            var dst = onnxModel.DrawOutput(ref src, output); 

            Cv2.ImShow("dd", dst);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();


            /* tact time 측정 */
            //float time = CheckTime(100, posePath, imgPath);
            //int fps = (int)(1000 / time);
        }

        static Mat RunYoloDetect(ref OnnxYolo onnxModel, ref Mat src, out int baseX1, out int baseY1)
        {
            var inputMat = onnxModel.MakeInputMat(ref src, out float ratio, out Point diff); 
            var results = onnxModel.ModelRun(ref inputMat);
            var output = onnxModel.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height);

            if (output.Count == 0)
            {
                baseX1 = baseY1 = 0;
                return src;
            }

            float bestProb = 0.0f;
            int bestIdx = 0;
            for (int i=0; i<output.Count; i++)
            {
                if (output[i][5] == 0 && output[i][4] > bestProb) // x1, y1, x2, y2, prob, clsidx
                {
                    bestIdx = i;
                }
            }

            var dst = onnxModel.MakeObjectCroppedMat(
                ref src,
                (int)output[bestIdx][0], 
                (int)output[bestIdx][1], 
                (int)output[bestIdx][2], 
                (int)output[bestIdx][3],
                out baseX1, out baseY1);
            return dst;
        }

        static float CheckTime(int cnt, string onnxPath, string imgPath)
        {
            OnnxModel onnxModel = new OnnxMMpose(onnxPath, 192, 256);
            Mat src = Cv2.ImRead(imgPath, ImreadModes.Color);
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < cnt; i++)
            {
                var inputMatTime = onnxModel.MakeInputMat(ref src, out float ratioTime, out Point diffTime);
                var resultsa = onnxModel.PostProcess(onnxModel.ModelRun(ref inputMatTime), ref ratioTime, ref diffTime, inputMatTime.Width, inputMatTime.Height); // 1(N) * 17(C) * 64(H) * 48(W)
            }
            stopwatch.Stop();

            return (stopwatch.ElapsedMilliseconds / cnt);
        }

    }
}