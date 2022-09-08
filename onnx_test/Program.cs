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
            const string imgPath = "C:\\Users\\Admin\\Documents\\20201211_General_118_DOC_A_M40_MM_024_0049.jpg";

            /* 모델 로드 */
            OnnxMMpose onnxModel = new OnnxMMpose(posePath, 192, 256);
            OnnxYolo yoloModel = new OnnxYolo(yoloPath, 640, 640);

            var croppedImg = RunYoloDetect(yoloModel, imgPath);


            /* Img(Mat) or ImgPath(String) -> InputMat 만들기 */
            //이미지 경로로 받기
            //var inputMat = onnxModel.MakeInputMat(imgPath, out Mat src, out float ratio, out Point diff);

            /* or (mat이 이미 있을때) */
            var inputMat = onnxModel.MakeInputMat(ref croppedImg, out float ratio, out Point diff);


            /* PostProcess까지 하면 결과값 출력 */
            /* KeyPoint 순서로 출력됨(OnnxMMpose.cs 참조) */
            var results = onnxModel.ModelRun(ref inputMat);
            var output = onnxModel.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height);

            /* 추가옵션(시각화) */
            var dst = onnxModel.DrawOutput(ref croppedImg, output); 

            Cv2.ImShow("dd", dst);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();


            /* tact time 측정 */
            //float time = CheckTime(100, posePath, imgPath);
            //int fps = (int)(1000 / time);
        }

        static Mat RunYoloDetect(OnnxYolo onnxModel, string imgPath)
        {
            var inputMat = onnxModel.MakeInputMat(imgPath, out Mat src, out float ratio, out Point diff); 
            var results = onnxModel.ModelRun(ref inputMat);
            var output = onnxModel.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height);

            if(output.Count == 0)
            {
                return src;
            }

            float bestProb = 0.0f;
            int bestIdx = 0;
            for (int i=0; i<output.Count; i++)
            {
                if (output[i][5] == 0 && output[i][4] > bestProb)
                {
                    bestIdx = i;
                }
            }

            var dst = onnxModel.MakeObjectCroppedMat(
                ref src,
                (int)output[bestIdx][0], 
                (int)output[bestIdx][1], 
                (int)output[bestIdx][2], 
                (int)output[bestIdx][3]);

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