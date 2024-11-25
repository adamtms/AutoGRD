# You need to have compiled the C++ code before running this script
#  g++ -O2 -std=c++11 -o orca.exe orca.cpp

import os
from glob import glob

from decorify import timeit

from RandomForestclassification import CreateClassifications
from Cooccurrence_computation import main as cooccurrence_computation
from Orca_algorithm import OperateOrca
from Embeeding_creation import getCorrelationMatrices, computeCorrelDist
from rank import main as rank
from xgboost_ranker import rank_families
from utils import YamlWriter, BaseExperiment


CreateClassifications = timeit(CreateClassifications)
cooccurrence_computation = timeit(cooccurrence_computation)
OperateOrca = timeit(OperateOrca)
getCorrelationMatrices = timeit(getCorrelationMatrices)
computeCorrelDist = timeit(computeCorrelDist)


class AutoGRDExeriment(BaseExperiment):
    def __init__(self):
        rank()
        super().__init__()


    def rank_families(self, _, dataset_name, __, seed, ___):
        return rank_families(dataset_name, seed)
    
    def perform_experiments(self, seeds=[0, 1, 2, 3, 4, 5], durations=...):
        with YamlWriter("durations_autoGRD") as writer:
            self.writer = writer
            return super().perform_experiments(seeds, [30])


    def _perform_experiment(self, seed, duration):
        self.main(seed)
        return super()._perform_experiment(seed, duration)


    def main(self, seed=0):
        # create folders
        writer = self.writer
        writer.increase_indent(seed)
        for folder in ["trees", "in_files", "orca", "in_files", "embedding"]:
            os.makedirs(folder, exist_ok=True)
    
        # create random forest classification for all datasets
        writer.increase_indent("RandomForest")
        datasets = sorted(glob("datasets/*.*"))
        for dataset in datasets:
            _, time = CreateClassifications(dataset, dataset.replace(".dat", ".csv").replace("datasets", "trees"))
            writer.add_partial_result(dataset.split("/")[-1].split(".")[0], time)
        writer.decrease_indent()

        # cooccurrence computation
        writer.increase_indent("Cooccurrence")
        paths = sorted(glob("trees/*.csv"))
        names = [path.replace("trees/", "").replace(".csv", "") for path in paths]
        for path, name in zip(paths, names):
            _, time = cooccurrence_computation([path], [name], "orca/", "in_files/")
            writer.add_partial_result(name, time)
        writer.decrease_indent()

        # orca algorithm
        writer.increase_indent("Orca")
        files = sorted(glob("in_files/*.in"))
        outputs = [x.replace("in_files", "orca") for x in files]
        for file, output in zip(files, outputs):
            _, time = OperateOrca(file, output)
            writer.add_partial_result(file.split("/")[-1].split(".")[0], time)
        writer.decrease_indent()

        # embedding creation
        writer.increase_indent("Embedding")
        paths = sorted(glob("orca/*.ndump2"))
        paths = [path.removesuffix(".in.ndump2") for path in paths]
        for path in paths:
            matrices, time_1 = getCorrelationMatrices([path], 16)
            _, time_2 = computeCorrelDist(matrices, None, None, None)
            writer.add_partial_result(path.split("/")[-1], time_1 + time_2)
        writer.decrease_indent()
        writer.decrease_indent()
                

if __name__ == "__main__":
    AutoGRDExeriment().perform_experiments(seeds=[0, 1], durations=[1])