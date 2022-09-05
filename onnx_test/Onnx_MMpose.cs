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
        private Size size;
        private Scalar backgroundColor;

        public Onnx_MMpose(string onnxPath, int width, int height, int color = 114)
        {
            if (string.IsNullOrWhiteSpace(onnxPath))
            {
                throw new ArgumentException("message", nameof(onnxPath));
            }

            var option = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
            };
            
            this.sess = new InferenceSession(onnxPath, option);
            this.size = new Size(width, height);
            this.backgroundColor = new Scalar(color, color, color);
        }
        
        public DisposableNamedOnnxValue[] ModelRun(List<NamedOnnxValue> input)
        {
            return this.sess.Run(input).ToArray();
        }


        public Mat MakeLetterBoxByMat(ref Mat input, out float ratio, out Point diff, out Point diff2, bool auto = true, bool scaleFill = false, bool isScaleUp = true)
        {
            Mat img = input.Clone();
            //Cv2.CvtColor(img, img, ColorConversionCodes.RGB2BGR);
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
    }
}
