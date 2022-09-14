using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace onnx_test
{
    internal class OnnxYolo : OnnxModel
    {
        private readonly float predThresh;
        private readonly float iouThresh;

        public OnnxYolo(string onnxPath, int width = 640, int height = 640, float predThresh = 0.2f, float iouThresh = 0.4f,  string inputName = "images")
            : base(onnxPath, width, height, inputName) {
            this.predThresh = predThresh;
            this.iouThresh = iouThresh;
        }

        public override List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, int imgWidth, int imgHeight, int bboxX = 0, int bboxY = 0, int batchIdx = 0)
        {
            var predValue = output[0].AsEnumerable<float>().ToArray();
            var predDims = output[0].AsTensor<float>().Dimensions.ToArray();

            var candidate = this.GetCandidate(predValue, predDims, batchIdx);
            Convert_xywh2xyxy(ref candidate);
            var refinedCandidate = this.GetDetectionMatrix(candidate);
            var nmsCandidate = this.NMS(refinedCandidate);
            return FitSizeofOutput(ref nmsCandidate, ref ratio, ref diff);
        }

        public override Mat DrawOutput(ref Mat inputMat, List<List<float>> output)
        {

            var outputMat = inputMat.Clone();

            for(int i = 0; i < output.Count; i++)
            {
                var x1 = (int)output[i][0];
                var y1 = (int)output[i][1];
                var x2 = (int)output[i][2];
                var y2 = (int)output[i][3];

                Cv2.Rectangle(outputMat, new Rect(x1, y1, x2 - x1, y2 - y1), new Scalar(255, 0, 0));
            }

            return outputMat;
        }

        protected override List<List<float>> FitSizeofOutput(ref List<List<float>> output, ref float ratio, ref Point diff)
        {
            var outputList = new List<List<float>>();

            for (int i = 0; i < output.Count; i++)
            {
                float fitX1 = Math.Max(output[i][0] - diff.X, 0) / ratio;
                float fitY1 = Math.Max(output[i][1] - diff.Y, 0) / ratio;
                float fitX2 = Math.Max(output[i][2] - diff.X, 0) / ratio;
                float fitY2 = Math.Max(output[i][3] - diff.Y, 0) / ratio;
                output[i][0] = fitX1;
                output[i][1] = fitY1;
                output[i][2] = fitX2;
                output[i][3] = fitY2;

                outputList.Add(output[i]);
            }

            return outputList;
        }
        private void Convert_xywh2xyxy(ref List<List<float>> output)
        {
            for (int i = 0; i < output.Count; i++)
            {
                var x1 = output[i][0] - output[i][2] / 2;
                var y1 = output[i][1] - output[i][3] / 2;
                var x2 = output[i][0] + output[i][2] / 2;
                var y2 = output[i][1] + output[i][3] / 2;
                output[i][0] = x1;
                output[i][1] = y1;  
                output[i][2] = x2;
                output[i][3] = y2;
            }
        }
        private List<List<float>> GetCandidate(float[] predValue, int[] predDims, int batchIdx)
        {
            var candidate = new List<List<float>>();

            for (int preds = 0; preds < predDims[1]; preds++)
            {
                int batch = batchIdx * predDims[1] * predDims[2];
                int predIdx = batch + preds * predDims[2];

                var objectPred = predValue[predIdx + 4];
                if (objectPred < this.predThresh) { continue; }

                var curCandidate = new List<float>();
                for (int i=0; i < predDims[2]; i++)
                {
                    curCandidate.Add(predValue[predIdx + i]); // x, y, w, h, preds, object preds...  
                }
                for (int i=5; i < curCandidate.Count; i++)
                {
                    curCandidate[i] *= curCandidate[4];
                }

                candidate.Add(curCandidate);
            }

            return candidate;
        }
        private List<List<float>> GetDetectionMatrix(List<List<float>> candidate, int max_nms = 30000)
        {
            var mat = new List<List<float>>();
            for (int i = 0; i < candidate.Count; i++)
            {
                int cls = -1;
                float max_score = 0;
                for (int j = 5; j < candidate[i].Count; j++)
                {
                    if (candidate[i][j] > this.predThresh && candidate[i][j] >= max_score)
                    {
                        cls = j;
                        max_score = candidate[i][j];
                    }
                }

                if (cls < 0) continue;

                List<float> tmpDetect = new List<float>();
                for (int j = 0; j < 4; j++) tmpDetect.Add(candidate[i][j]); //box
                tmpDetect.Add(candidate[i][cls]);   //class prob
                tmpDetect.Add(cls - 5);             //class
                mat.Add(tmpDetect);
            }

            //max_nms sort
            mat.Sort((a, b) => (a[4] > b[4]) ? -1 : 1);

            if (mat.Count > max_nms)
            {
                mat.RemoveRange(max_nms, mat.Count - max_nms);
            }
            return mat;
        }
        private List<List<float>> NMS(List<List<float>> candidate, int maxWH = 4096)
        {
            List<List<float>> nmsCandidate = new List<List<float>>();

            List<Rect> bboxes = new List<Rect>();
            List<float> confidences = new List<float>();

            for (int i = 0; i < candidate.Count; i++)
            {
                var diff_class = (int)(maxWH * candidate[i][5]);

                Rect box = new Rect((int)candidate[i][0] + diff_class, (int)candidate[i][1] + diff_class,
                    (int)(candidate[i][2] - candidate[i][0]), (int)(candidate[i][3] - candidate[i][1]));
                bboxes.Add(box);
                confidences.Add(candidate[i][4]);
            }
            CvDnn.NMSBoxes(bboxes, confidences, this.predThresh, this.iouThresh, out int[] indices);

            for (int i=0; i<indices.Length; i++) { nmsCandidate.Add(candidate[indices[i]]); }

            return nmsCandidate;
        }

        public Mat MakeObjectCroppedMat(ref Mat src, int x1, int y1, int x2, int y2, out int baseX1, out int baseY1, int padding = 50)
        {
            baseX1 = x1 = Math.Max(0, x1 - padding);
            baseY1 = y1 = Math.Max(0, y1 - padding);
            x2 = Math.Min(src.Width, x2 + padding);
            y2 = Math.Min(src.Height, y2 + padding);

            Rect rect = new Rect(x1, y1, x2 - x1, y2 - y1);
            Mat output = src.SubMat(rect);

            return output;
        }
    }
}
