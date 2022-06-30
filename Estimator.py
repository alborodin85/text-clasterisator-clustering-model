from sklearn import metrics
import numpy


class Estimator:
    @staticmethod
    def estimate(n_clusters: int, predictions: numpy.ndarray, targets: numpy.ndarray, train: numpy.ndarray) -> list:
        """
        :param n_clusters: число кластеров
        :param predictions: список предсказанных кластеров
        :param targets: список фактических кластеров
        :param train: numpy-массив с тренировочными данными
        :return: словарь с усредненными метриками и словарь с соотношениями кластеров predictions и targets
        """
        [predictClusters, targetClusters] = Estimator.groupLabels(n_clusters, predictions, targets)
        matchedClusters = Estimator.matchClusters(n_clusters, predictClusters, targetClusters)
        medResult = Estimator.estimateByParams(matchedClusters, n_clusters, train, predictions)

        return [medResult, matchedClusters]

    @staticmethod
    def estimateByParams(matchedClusters, n_clusters, train, predictions):
        sumPrecision = 0
        sumRecall = 0
        sumF1 = 0
        for accItem in matchedClusters:
            sumPrecision += accItem['Precision']
            sumRecall += accItem['Recall']
            sumF1 += accItem['F1']

        medResult = {
            'Precision': sumPrecision / n_clusters,
            'Recall': sumRecall / n_clusters,
            'F1': sumF1 / n_clusters,
        }

        X = train
        labels = predictions
        silhouette = metrics.silhouette_score(X=X, labels=labels)

        medResult['silhouette'] = silhouette

        return medResult

    @staticmethod
    def groupLabels(n_clusters, predictions, targets):
        predictClusters = [[] for i in range(n_clusters)]
        for predictId in range(len(predictions)):
            predictClusters[predictions[predictId]].append(predictId)

        targetClusters = [[] for i in range(n_clusters)]
        for targetId in range(len(targets)):
            targetClusters[targets[targetId]].append(targetId)

        return [predictClusters, targetClusters]

    @staticmethod
    def matchClusters(n_clusters, predictClusters, targetClusters):
        result = []
        for predictClaster in range(n_clusters):
            predictItems = predictClusters[predictClaster]
            maxAccuracy = {
                'predict_cluster': None,
                'target_cluster': None,
                'Precision': 0,
                'Recall': 0,
                'F1': 0,
            }
            for targetCluster in range(n_clusters):
                targetItems = targetClusters[targetCluster]
                TP = 0  # Истинно-положительное
                FP = 0  # Ложно-положительное
                FN = 0  # Которые не найдены
                for predictItem in predictItems:
                    if predictItem in targetItems:
                        TP += 1
                    else:
                        FP += 1
                for targetItem in targetItems:
                    if targetItem not in predictItems:
                        FN += 1

                Precision = 0
                if (TP + FP) != 0:
                    Precision = TP / (TP + FP)
                Recall = 0
                if (TP + FN) != 0:
                    Recall = TP / (TP + FN)
                F1 = 0
                if Precision != 0 or Recall != 0:
                    F1 = 2 * (Precision * Recall) / (Precision + Recall)
                if F1 > maxAccuracy['F1']:
                    maxAccuracy = {
                        'predict_cluster': predictClaster,
                        'target_cluster': targetCluster,
                        'Precision': Precision,
                        'Recall': Recall,
                        'F1': F1,
                    }
            result.append(maxAccuracy)

        return result
