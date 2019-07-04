using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ML
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=============== 输入训练数据文件路径 ===============");

            string InputDataPath = Console.ReadLine();

            Console.WriteLine("=============== 正在生成临时训练和测试数据集 ===============");
            DataPrepare(InputDataPath);

            string TrainDataPath = Path.GetDirectoryName(InputDataPath) + "\\Ratings_Train.csv";
            string TestDataPath = Path.GetDirectoryName(InputDataPath) + "\\Ratings_Test.csv";
            StartMachineLearning(TrainDataPath, TestDataPath).Wait();

            Console.ReadKey();

            try
            {
                if (File.Exists(TrainDataPath))
                {
                    File.Delete(TrainDataPath);
                }

                if (File.Exists(TestDataPath))
                {
                    File.Delete(TestDataPath);
                }

                Console.WriteLine();
                Console.WriteLine("=============== 已清理临时训练数据和临时测试数据，按任意键关闭 ===============");
            }
            catch (IOException)
            {
                Console.WriteLine("=============== 由于文件被其他程序占用，无法清理临时文件 ===============");
            }

            Console.ReadKey();
        }

        private static async Task StartMachineLearning(string TrainDataPath, string TestDataPath)
        {
            if (File.Exists(TrainDataPath) && File.Exists(TestDataPath))
            {
                var BestModel = await TrainAndGetBestModel(TrainDataPath);

                Console.WriteLine("=============== 模型训练完成 ===============");

                Console.WriteLine("=============== 正在读取测试数据文件 ===============");

                IDataView TestDataView = MLCProvider.Current.Data.LoadFromTextFile<BookRating>(TestDataPath, ',', true);

                EvaluateModel(TestDataView, BestModel);

                Console.WriteLine("=============== 请输入UserID和ISBN以预测Rating ===============");

                FLAG1:
                Console.WriteLine("UserID:");
                string UserID = Console.ReadLine();
                if(string.IsNullOrWhiteSpace(UserID))
                {
                    Console.WriteLine("此项必填，不可为空");
                    Console.WriteLine();
                    goto FLAG1;
                }
                Console.WriteLine();

                FLAG2:
                Console.WriteLine("Age:");
                string Age = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(Age))
                {
                    Console.WriteLine("此项必填，不可为空");
                    Console.WriteLine();
                    goto FLAG2;
                }
                Console.WriteLine();

                FLAG3:
                Console.WriteLine("ISBN:");
                string ISBN = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(Age))
                {
                    Console.WriteLine("此项必填，不可为空");
                    Console.WriteLine();
                    goto FLAG3;
                }
                Console.WriteLine();

                using (PredictionEngine<BookRating, BookRatingPrediction> Engine = MLCProvider.Current.Model.CreatePredictionEngine<BookRating, BookRatingPrediction>(BestModel))
                {
                    BookRating PredictInput = new BookRating
                    {
                        UserId = UserID,
                        ISBN = ISBN,
                        Age = Age
                    };
                    BookRatingPrediction PredictResult = Engine.Predict(PredictInput);
                    Console.WriteLine("预测结果如下：" + (PredictResult.PredictedLabel ? "推荐" : "不推荐") + "，预测该书籍评分:" + Sigmoid(PredictResult.Score).ToString("0.##"));
                }
            }
            else
            {
                Console.WriteLine("=============== 数据文件路径无效 ===============");
            }
        }

        private static Task<ITransformer> TrainAndGetBestModel(string FilePath)
        {
            return Task.Factory.StartNew(() =>
            {
                MLContext MLC = MLCProvider.Current;

                IDataView TrainingDataView = MLC.Data.LoadFromTextFile<BookRating>(FilePath, ',', true);
                TrainingDataView = MLC.Data.Cache(TrainingDataView);

                Console.WriteLine("=============== 正在读取训练数据文件 ===============");

                EstimatorChain<ColumnConcatenatingTransformer> DataPipeLine = MLC.Transforms.Text.FeaturizeText("UserIdFeaturized", nameof(BookRating.UserId))
                .Append(MLC.Transforms.Text.FeaturizeText("ISBNFeaturized", nameof(BookRating.ISBN)))
                .Append(MLC.Transforms.Text.FeaturizeText("AgeFeaturized", nameof(BookRating.Age)))
                .Append(MLC.Transforms.Concatenate("Features", "UserIdFeaturized", "ISBNFeaturized", "AgeFeaturized"));

                Console.WriteLine("=============== 正在使用交叉验证训练预测模型 ===============");


                FieldAwareFactorizationMachineTrainer.Options Options = new FieldAwareFactorizationMachineTrainer.Options
                {
                    Verbose = true,
                    NumberOfIterations = 10,
                    FeatureColumnName = "Features",
                    Shuffle = true
                };

                EstimatorChain<FieldAwareFactorizationMachinePredictionTransformer> TrainingPipeLine = DataPipeLine.Append(MLC.BinaryClassification.Trainers.FieldAwareFactorizationMachine(Options));

                var CVResult = MLC.BinaryClassification.CrossValidate(TrainingDataView, TrainingPipeLine);

                return CVResult.OrderByDescending(t => t.Metrics.Accuracy).Select(r => r.Model).FirstOrDefault();

            }, TaskCreationOptions.LongRunning);
        }

        public static void EvaluateModel(IDataView TestDataView, ITransformer model)
        {
            Console.WriteLine("=============== 正在评估模型拟合效果 ===============");

            CalibratedBinaryClassificationMetrics Metrics = MLCProvider.Current.BinaryClassification.Evaluate(model.Transform(TestDataView), "Label", "Score", predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine("=============== 评估结果如下 ===============");

            Console.WriteLine();
            Console.WriteLine("准确度: " + Metrics.Accuracy);
            Console.WriteLine();
        }

        public static float Sigmoid(float x)
        {
            return (float)(10 / (1 + Math.Exp(-x)));
        }

        public static void DataPrepare(string InputFilePath)
        {
            try
            {
                string[] dataset = File.ReadAllLines(InputFilePath);

                string[] new_dataset = new string[dataset.Length];
                new_dataset[0] = dataset[0];
                for (int i = 1; i < dataset.Length; i++)
                {
                    string line = dataset[i];
                    string[] lineSplit = line.Split(',');
                    double rating = double.Parse(lineSplit[2]);
                    rating = rating > 6 ? 1 : 0;
                    lineSplit[2] = rating.ToString();
                    string new_line = string.Join(",", lineSplit);
                    new_dataset[i] = new_line;
                }
                dataset = new_dataset;
                int numLines = dataset.Length;
                var body = dataset.Skip(1);
                var sorted = body.Select(line => new { SortKey = int.Parse(line.Split(',')[0]), Line = line })
                                 .OrderBy(x => x.SortKey)
                                 .Select(x => x.Line);

                string TrainPath = Path.GetDirectoryName(InputFilePath) + "\\Ratings_Train.csv";
                string TestPath = Path.GetDirectoryName(InputFilePath) + "\\Ratings_Test.csv";

                if (File.Exists(TrainPath))
                {
                    File.Delete(TrainPath);
                }

                if (File.Exists(TestPath))
                {
                    File.Delete(TestPath);
                }

                File.WriteAllLines(TrainPath, dataset.Take(1).Concat(sorted.Take((int)(numLines * 0.8))));

                File.WriteAllLines(TestPath, dataset.Take(1).Concat(sorted.TakeLast((int)(numLines * 0.2))));
            }
            catch (IOException)
            {
                Console.WriteLine("=============== 由于文件被其他程序占用，无法解析数据文件内容 ===============");
            }
        }
    }
}
