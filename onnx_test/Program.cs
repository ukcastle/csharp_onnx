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
            Onnx_MMpose onnxPose = new Onnx_MMpose("C:\\Users\\Admin\\Documents\\output.onnx", 192, 256);

            //Mat src = Cv2.ImRead("C:\\Users\\Admin\\Documents\\3326.jpg", ImreadModes.Color);
            //var inputMat = onnxPose.MakeInputMat(ref src, out float ratio, out Point diff, out Point diff2, auto:false, scaleFill:false);

            var inputMat = onnxPose.MakeInputMat("C:\\Users\\Admin\\Documents\\20201123_General_001_DIS_S_F20_SS_001_0001.jpg", out Mat src, out float ratio, out Point diff, out Point diff2);
            var results = onnxPose.ModelRun(ref inputMat); // 1(N) * 17(C) * 64(H) * 48(W)
            var output = onnxPose.PostProcess(results, ref ratio, ref diff, ref diff2, inputMat.Width, inputMat.Height);

            var dst = onnxPose.DrawOutput(ref src, output);
            
            Cv2.ImShow("dd", dst);
            Cv2.WaitKey();
        }

    }
}