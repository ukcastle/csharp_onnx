using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace onnx_test
{
    internal class OnnxYolo : OnnxModel
    {
        private readonly float predThresh;
        private readonly float iouThresh;


        public OnnxYolo(string onnxPath, float predThresh = 0.2f, float iouThresh = 0.4f, int width = 192, int height = 256, string inputName = "input.1")
            : base(onnxPath, width, height, inputName) {
            this.predThresh = predThresh;
            this.iouThresh = iouThresh;
        }
        public override List<List<float>> PostProcess(
            DisposableNamedOnnxValue[] output, ref float ratio, ref Point diff, int imgWidth, int imgHeight, int bboxX = 0, int bboxY = 0, int batchIdx = 0)
        {
            var predValue = output[0].AsEnumerable<float>().ToArray();
            var predDims = output[0].AsTensor<float>().Dimensions.ToArray();

            var candidate = new List<List<float>>(); 

            for (int preds = 0; preds < predDims[1]; preds++)
            {
                int batch = batchIdx * predDims[1];
                int predIdx = batch + preds * predDims[2];

                var objectPred = predValue[predIdx + 4];
                if(objectPred < this.predThresh) { continue; }

                var curCandidate = new List<float>();
                for (int i=0;i< 4; i++)
                {
                    curCandidate.Add(predValue[predIdx+i]); // input x y w h  
                }

                curCandidate.Add(objectPred); // pred

                for (int i=5; i < predDims[2]; i++)
                {
                    curCandidate.Add(predValue[predIdx+i] * objectPred);
                }
                candidate.Add(curCandidate);
            }


            //return this.FitSizeofOutput(ref keyPoints, ref ratio, ref diff);
        }


        private List<List<float>> GetCandidate(float[] pred, int[] pred_dim)
        {
            List<List<float>> candidate = new List<List<float>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;  // objectness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];
                    if (value > this.predThresh)
                    {
                        List<float> tmp_value = new List<float>();
                        for (int i = 0; i < pred_dim[2]; i++)
                        {
                            int sub_idx = idx1 + i;
                            tmp_value.Add(pred[sub_idx]);
                        }
                        candidate.Add(tmp_value);
                    }
                }
            }
            return candidate;
        }
    }
}
