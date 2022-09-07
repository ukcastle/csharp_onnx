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
            const string onnxPath = "C:\\Users\\Admin\\Documents\\hr.onnx";
            const string yoloPath = "C:\\Users\\Admin\\Documents\\best.onnx";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\3326.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\20201123_General_001_DIS_S_F20_SS_001_0001.jpg";
            const string imgPath = "C:\\Users\\Admin\\Documents\\20201202_General_068_DOS_A_M40_MM_007_0053.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\aa1.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\1122.jpg";


            OnnxYolo onnxYolo = new OnnxYolo(yoloPath);
            var inputMat = onnxYolo.MakeInputMat(imgPath, out Mat src, out float ratio, out Point diff);
            var results = onnxYolo.ModelRun(ref inputMat);
            var output = onnxYolo.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height);
            var dst = onnxYolo.DrawOutput(ref src, output);


            //float time = CheckTime(100, onnxPath, imgPath);
            //int fps = (int)(1000 / time);

            //OnnxMMpose onnxPose = new OnnxMMpose(onnxPath, 192, 256);

            //var inputMat = onnxPose.MakeInputMat(imgPath, out Mat src, out float ratio, out Point diff);
            //var results = onnxPose.ModelRun(ref inputMat); // 1(N) * 17(C) * 64(H) * 48(W)
            //var output = onnxPose.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height); // 17(Key Point) * 3 (x, y, pred)

            //var dst = onnxPose.DrawOutput(ref src, output);

            Cv2.ImShow("dd", dst);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();

        }

        static float CheckTime(int cnt, string onnxPath, string imgPath)
        {
            OnnxMMpose onnxPose = new OnnxMMpose(onnxPath, 192, 256);

            Mat src = Cv2.ImRead(imgPath, ImreadModes.Color);
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < cnt; i++)
            {
                var inputMatTime = onnxPose.MakeInputMat(ref src, out float ratioTime, out Point diffTime);
                var resultsa = onnxPose.PostProcess(onnxPose.ModelRun(ref inputMatTime), ref ratioTime, ref diffTime, inputMatTime.Width, inputMatTime.Height); // 1(N) * 17(C) * 64(H) * 48(W)
            }
            stopwatch.Stop();

            return (stopwatch.ElapsedMilliseconds / cnt);
        }

    }
}