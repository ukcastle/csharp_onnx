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
        /*
         * 100회 기준 tact time
         * hr : 97ms, 10fps / 33ms, 30fps
         * res50fp16 : 65ms, 15fps / 26ms, 38fps
         * mobv2 : 48ms, 20fps /  14ms, 71fps
         * shufflev2 : 33ms, 30fps / 16ms, 62fps
         */
         
        [STAThread]
        static void Main()
       {
            const string onnxPath = "C:\\Users\\Admin\\Documents\\hr.onnx";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\3326.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\20201123_General_001_DIS_S_F20_SS_001_0001.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\aa1.jpg";
            const string imgPath = "C:\\Users\\Admin\\Documents\\Capture_213.jpg";
            //const string imgPath = "C:\\Users\\Admin\\Documents\\1122.jpg";

            //float time = CheckTime(100, onnxPath, imgPath);
            //int fps = (int)(1000 / time);

            OnnxMMpose onnxPose = new OnnxMMpose(onnxPath, 192, 256);
            
            var inputMat = onnxPose.MakeInputMat(imgPath, out Mat src, out float ratio, out Point diff);
            var results = onnxPose.ModelRun(ref inputMat); // 1(N) * 17(C) * 64(H) * 48(W)
            var output = onnxPose.PostProcess(results, ref ratio, ref diff, inputMat.Width, inputMat.Height); // 17(Key Point) * 3 (x, y, pred)

            var dst = onnxPose.DrawOutput(ref src, output);
            
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