using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnx_test
{
    internal abstract class Onnx_Model
    {
        // Member
        protected readonly InferenceSession sess;
        protected readonly Size size;
        protected readonly Scalar backgroundColor;
        protected readonly string inputName;
        protected static readonly float[,] normalizeValue = new float[2, 3]
        {
            {0.406f, 0.456f, 0.485f}, // mean b g r
            {0.225f, 0.224f, 0.229f} // std b g r
            //{0.485f, 0.456f, 0.406f}, // mean r g b
            //{0.229f, 0.224f, 0.225f} // std r g b
        };

        // Constructor
        protected Onnx_Model(string onnxPath, int width, int height, string inputName, int backgroundColor = 114)
        {
            var option = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                //GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC, // 조금 더 걸림
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            };

            this.sess = new InferenceSession(onnxPath, option);
            this.size = new Size(width, height);
            this.inputName = inputName;
            this.backgroundColor = new Scalar(backgroundColor, backgroundColor, backgroundColor);
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
            diff = new Point(halfBackgroundWidth + ifBackgroundWidthIsOdd, halfBackgroundHeigth + ifBackgroundHeightIsOdd);
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
                NamedOnnxValue.CreateFromTensor(this.inputName, new DenseTensor<float>(Onnx_Model.Mat2Array(mat), new[] { 1, 3, mat.Height, mat.Width }))
            };
            return this.sess.Run(onnxInput).ToArray();
        }

        public abstract List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, int imgWidth, int imgHeight, int bboxX, int bboxY, int batchIdx);

        // Private Func
        protected static void RefinePointBySize(ref List<float> keyValue, int scaleX, int scaleY, int bboxX, int bboxY)
        {
            keyValue[0] = keyValue[0] * scaleX + bboxX; // x
            keyValue[1] = keyValue[1] * scaleY + bboxY; // y
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
                        array[convertedIdx] = (matPointer[baseIdx] - Onnx_Model.normalizeValue[0, c]) / Onnx_Model.normalizeValue[1, c];
                    }
                }
            }
            return array;
        }
    }
}
