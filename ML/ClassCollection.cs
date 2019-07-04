using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML
{
    public sealed class BookRating
    {
        [LoadColumn(0)]
        public string UserId;
        [LoadColumn(1)]
        public string ISBN;
        [LoadColumn(2)]
        public bool Label;
        [LoadColumn(8)]
        public string Age;
    }

    public sealed class BookRatingPrediction
    {
        public bool PredictedLabel;

        public float Score;
    }

    public sealed class MLCProvider
    {
        public static MLContext Current
        {
            get
            {
                return Provider ?? (Provider = new MLContext());
            }
        }

        private static MLContext Provider;
    }

    public static class Extensions
    {
        public static IEnumerable<T> TakeLast<T>(this IEnumerable<T> source, int N)
        {
            return source.Skip(Math.Max(0, source.Count() - N));
        }
    }
}
