%% Sign Alphabet Classification

clearvars
clear all
clc

%% INICIALIZACIÓN, CARGA DEL DATASET

% Se establece la direccion del dataset
dir = './Gesture Image Data/';

% Se carga el dataset completo
allData = imageDatastore(dir, 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

% Se seleccionan los datos a clasificar
[Data, noData] = splitEachLabel(allData,0.1,'Randomized');
% Data = allData;
[trainingSet, testSet] = splitEachLabel(Data,0.98,'Randomized'); % training set es el de prueba y el test set es el de entrenamiento

% Se muestran los datos a clasificar
countEachLabel(trainingSet);
countEachLabel(testSet);

% Se toma imagen de prueba para sacar tamaño de la matrices
img = readimage(testSet, 1);

% Extracción de características para entrenamiento
numImages = numel(trainingSet.Files);

trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;

Y = [trainingLabels];

%% DESCRIPTORES
X = helperExtractFeatures(trainingSet);

%% CARGA DEL ÁRBOL DE DECISIÓN

% Mdl = fitctree(X,Y,'CrossVal','on');
Mdl = fitctree(X,Y);
Mdl_c = compact(Mdl);

%% DESCRIPTORES DATA TEST
Xtest = helperExtractFeatures(testSet);

%%
predictedLabels = predict(Mdl_c, Xtest);

figure(1)
confusionchart(testLabels,predictedLabels,...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');
