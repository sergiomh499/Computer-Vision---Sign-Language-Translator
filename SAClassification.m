%% Sign Alphabet Classification
clear all
clc
%% Inicializacion
disp('Inicializacion')
tic
% Se establece la direccion del dataset
dir = './Gesture Image Data/';

% Se carga el dataset completo
allData = imageDatastore(dir, 'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

% Se seleccionan los datos a clasificar
% [Data, noData] = splitEachLabel(allData,1,'Randomized');
Data = allData;
[trainingSet, testSet] = splitEachLabel(Data,0.6,'Randomized');

% Se muestran los datos a clasificar
countEachLabel(trainingSet)
countEachLabel(testSet)

% Se toma imagen de prueba para sacar tamaño de la matrices
img = readimage(trainingSet, 1);
img_bw = rgb2gray(img);
img_bin = imbinarize(img_bw);

% Tamaño de la celda para HOG
cellSize = [10 10];

% Calculamos tamaño de las caractericticas de HOG
[hog, vis] = extractHOGFeatures(img,'CellSize', cellSize);
hogFeatureSize = length(hog);

toc

%% ENTRENAMIENTO

% Extracción de características para entrenamiento
disp('Extraccion de caracteristicas para entrenamiento')
tic

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    img_bw = rgb2gray(img);
    img_bin = imbinarize(img_bw);
    img(:,:,1) = abs(1 - double(img_bin)).*double(img(:,:,1));
    img(:,:,2) = abs(1 - double(img_bin)).*double(img(:,:,2));
    img(:,:,3) = abs(1 - double(img_bin)).*double(img(:,:,3));
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end
toc

% Get labels for each image.
disp('Obtener etiquetas de imagenes')
tic
trainingLabels = trainingSet.Labels;
toc

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
disp('Entrenamiento')
tic
classifier = fitcecoc(trainingFeatures, trainingLabels);
toc

%% CLASIFICACIÓN
% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
disp('Extraccion de caracteristicas para clasificacion')
tic
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(...
    testSet, hogFeatureSize, cellSize);
toc

% Make class predictions using the test features.
disp('Clasificacion')
tic
predictedLabels = predict(classifier, testFeatures);

figure(1)
confusionchart(testLabels,predictedLabels,...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% TESTEO DE DETECCIÓN DE IMAGENES

% for k = 1:50
%     i = randi([1 55500]);
%     
%     imgFeatures  = zeros(1, hogFeatureSize, 'single');
%     
%     img = readimage(allData, i);
%     imgFeatures(1, :) = extractHOGFeatures(img, 'CellSize', cellSize);
%     
%     testLabels = allData.Labels;
%     testLabel = testLabels(i);
%     
%     predictedLabels = predict(classifier, imgFeatures);
%     
%     figure(2)
%     subplot(5,10,k)
%     imshow(img)
%     title(['Entrada: ' testLabel ' Prediccion: ' predictedLabels])
%     
%     TESTLABELS(k) = testLabel;
%     PREDICTEDLABELS(k) = predictedLabels;
% end
% 
% figure(3)
% confusionchart(TESTLABELS,PREDICTEDLABELS,...
%     'RowSummary','row-normalized','ColumnSummary','column-normalized');

%% DICCIONARIO DE SIGNOS
% dataLabels = allData.Labels;
% figure(3)
% for i = 1:37
%     im = readimage(allData,i+1500*(i-1));
%     subplot(4,10,i)
%     imshow(im)
%     title(dataLabels(i+1500*(i-1)))
% end

%% DETECTAR POR WEBCAM
% clear('cam')
% cam = webcam;
% final = 0;
% snap = 0;
% while final == 0
% snap = input('¿Tomar captura? Sí[1] No[2]: ');
% 
% if snap == 1
%     preview(cam);
%     pause(5)
%     closePreview(cam);
%     img = snapshot(cam);
%     targetSize = [size(img,1) size(img,1)];
%     r = centerCropWindow2d(size(img),targetSize);
%     imgLQ = imcrop(img,r);
%     imgLQ = imresize(imgLQ, [50 50]);
%     img_bw = rgb2gray(imgLQ);
%     img_bin = imbinarize(img_bw);
%     imgLQ(:,:,1) = abs(1 - double(img_bin)).*double(imgLQ(:,:,1));
%     imgLQ(:,:,2) = abs(1 - double(img_bin)).*double(imgLQ(:,:,2));
%     imgLQ(:,:,3) = abs(1 - double(img_bin)).*double(imgLQ(:,:,3));
%     imgFeature(1, :) = extractHOGFeatures(imgLQ, 'CellSize', cellSize);
%     predictedLabels = predict(classifier, imgFeature);
%     
%     figure
%     imshow(imgLQ)
%     title(predictedLabels)
%     
% elseif snap == 2
%     final = 1; 
% end
% 
% end
% clear('cam')

