clear all
close all
clc

load('Your latent Variabls')


%% Normalization
[latentYoloTr, muYolo, sigYolo] = zscore(latentYoloTr);
[latentDenseTr, muDense, sigDense] = zscore(latentDenseTr);
[latentVGGTr, muVGG, sigVGG] = zscore(latentVGGTr);

latentYoloVa = (latentYoloVa - muYolo)./sigYolo;
latentDenseVa = (latentDenseVa - muDense)./sigDense;
latentVGGVa = (latentVGGVa - muVGG)./sigVGG;

TotalFeatures = zscore(TotalFeatures);

%% Radiomics
glmnetOptions = glmnetSet;
glmnetOptions.alpha = 1;
glmnetOptions.maxit = 1e+07;
featureSelected = [];

c = cvpartition(survival(:,2),'holdout',0.3);

TrainingFeatures = TotalFeatures(c.training,:);
TrainingSurvival = survival(c.training,:);
TestFeatures = TotalFeatures(c.test,:);
TestSurvival = survival(c.test,:);

featureSelected = find(lassoMdl.glmnet_fit.beta(:,find(lassoMdl.lambda == lassoMdl.lambda_min)));


FeatureNames(featureSelected,:)
cvglmnetPlot(lassoMdl)

scoreTrain = cvglmnetPredict(lassoMdl,TotalFeatures(c.training,:),'lambda_min');
scoreTest = cvglmnetPredict(lassoMdl,TotalFeatures(c.test,:),'lambda_min');

figHandler(1) = figure;
[~, ~, HRTrainRad, HRciTrainRad, ~, ~, ~, pTrainRad, ~] = ...
    logrank([TrainingSurvival(find(scoreTrain >= median(scoreTrain)),1),1-TrainingSurvival(find(scoreTrain >= median(scoreTrain)),2)], ...
    [TrainingSurvival(find(scoreTrain < median(scoreTrain)),1), 1-TrainingSurvival(find(scoreTrain < median(scoreTrain)),2)]);

figHandler(2) = figure;
[~, ~, HRTestRad, HRciTestRad, ~, ~, ~, pTestRad, ~] = ...
    logrank([TestSurvival(find(scoreTest >= median(scoreTrain)),1),1-TestSurvival(find(scoreTest >= median(scoreTrain)),2)], ...
    [TestSurvival(find(scoreTest < median(scoreTrain)),1), 1-TestSurvival(find(scoreTest < median(scoreTrain)),2)]);


%% Yolo
YoloFeaturesFinal = latentYoloTr;
TrainingYoloFeatures = YoloFeaturesFinal(c.training,:);
TestYoloFeatures = YoloFeaturesFinal(c.test,:);

[r,p] = corr(TrainingYoloFeatures,TrainingFeatures);
p = p*size(TrainingYoloFeatures,2);

rSortYolo = sort(abs(r(:)),'descend');
pSortYolo = sort(p(:),'ascend');

rThYolo = rSortYolo(sum(rSortYolo>.4))
pthYolo = pSortYolo(sum(rSortYolo>.4))

significantYoloFeatures = [];
i = 1;
while i <= size(r,1)
    if ~isempty(find(abs(r(i,:))>rThYolo))
        significantYoloFeatures = [significantYoloFeatures; i];
    end
    i = i + 1;
end
size(significantYoloFeatures)

TrainingYoloFeatures = TrainingYoloFeatures(:,significantYoloFeatures);
TestYoloFeatures = TestYoloFeatures(:,significantYoloFeatures);
ValidationYoloFeatures = latentYoloVa(:,significantYoloFeatures);

glmnetOptions.alpha = 1;
lasoYoloMdl = cvglmnet(TrainingYoloFeatures,TrainingSurvival,'cox',glmnetOptions,[],5);
featureSelectedYolo = find(lasoYoloMdl.glmnet_fit.beta(:,find(lasoYoloMdl.lambda == lasoYoloMdl.lambda_min)));

significantYoloFeatures(featureSelectedYolo)
cvglmnetPlot(lasoYoloMdl)

scoreTrainYolo = cvglmnetPredict(lasoYoloMdl,TrainingYoloFeatures,'lambda_min');
scoreTestYolo = cvglmnetPredict(lasoYoloMdl,TestYoloFeatures,'lambda_min');
scoreValidYolo = cvglmnetPredict(lasoYoloMdl,ValidationYoloFeatures,'lambda_min');

figHandler(3) = figure;
[~, ~, HRTrainYolo, HRciTrainYolo, ~, ~, ~, pTrainYolo, ~] = ...
    logrank([TrainingSurvival(find(scoreTrainYolo >= median(scoreTrainYolo)),1),1-TrainingSurvival(find(scoreTrainYolo >= median(scoreTrainYolo)),2)], ...
    [TrainingSurvival(find(scoreTrainYolo < median(scoreTrainYolo)),1), 1-TrainingSurvival(find(scoreTrainYolo < median(scoreTrainYolo)),2)]);

figHandler(4) = figure;
[~, ~, HRTestYolo, HRciTestYolo, ~, ~, ~, pTestYolo, ~] = ...
    logrank([TestSurvival(find(scoreTestYolo >= median(scoreTrainYolo)),1),1-TestSurvival(find(scoreTestYolo >= median(scoreTrainYolo)),2)], ...
    [TestSurvival(find(scoreTestYolo < median(scoreTrainYolo)),1), 1-TestSurvival(find(scoreTestYolo < median(scoreTrainYolo)),2)]);

figHandler(5) = figure;
[~, ~, HRValidYolo, HRciValidYolo, ~, ~, ~, pValidYolo, ~] = ...
    logrank([ValidSurvival(find(scoreValidYolo >= median(scoreTrainYolo)),1),1-ValidSurvival(find(scoreValidYolo >= median(scoreTrainYolo)),2)], ...
    [ValidSurvival(find(scoreValidYolo < median(scoreTrainYolo)),1), 1-ValidSurvival(find(scoreValidYolo < median(scoreTrainYolo)),2)]);
