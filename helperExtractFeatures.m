function X = helperExtractFeatures(dataSet)

numImages = numel(dataSet.Files);

for i = 1:numImages
    img = readimage(dataSet, i);
    % -------------------------------------------------
    %  ----------ORIENTACIÓN DE LA IMAGEN--------------
    % -------------------------------------------------
    % Conversión de la imagen a escala de grises
    B=rgb2gray(img);
    
    % Binarización de la imagen y aplicación del algoritmo Hough
    im4b3n = (1-imbinarize(B));
    m_AC = hough(im4b3n);
    
    % Picos de la transformada de Hough aplicada a la imagen
    p = houghpeaks(m_AC,20);
    m = mean(p(:,2));
    
    % Imagen rotada según la media de los picos
    im4r = imrotate (B, m);
    
    % Clasificación de la imagen en horizontal o vertical
    if (m>=45 && m<=135) || (m>=225 && m<=315)
        S = 1; % 1 -> Vertical
    else
        S = 0; % 0 -> Horizontal
    end
    
    % -------------------------------------------------
    %  ---------------GRUPOS DE DEDOS------------------
    % -------------------------------------------------
    
    % Binarización de la imagen y aplicación de filtro
    im4r_binarized = not(imbinarize(B));
    de = strel('disk',3);
    afterOpening = imopen(im4r_binarized,de);
    
    K = imfill(afterOpening,'holes');
    
    im_dedos = im4r_binarized - K;
    cc = bwconncomp(im_dedos);
    
    % Extracción de las areas
    areas = regionprops(cc, 'Area', 'Eccentricity');
    
    % Extracción del numero de areas
    L = length(areas);
    dedo = 0;
    
    % Comprobación del tamaño de las areas para saber si es un dedo
    for j=1:L
        if areas(j).Area > 4
            dedo = dedo + 1;
        end
    end
    
    % -------------------------------------------------
    %  ----DISCONTINUIDADES EN LA REGIÓN CENTRAL-------
    % -------------------------------------------------
        
    A_grey=imadjust(B);
    
    % Extraemos los bordes
    A1=edge(A_grey,'canny');
    A2=not(imbinarize(A_grey));
    
    % Aplicacion de operaciones morfologicas
    A3=imfill(A2,'holes');
    se = strel('sphere',2);
    J=imerode(A3,se);
    Aint=A1.*J;
    
    des = sum(sum(double(Aint)));
    
    % -------------------------------------------------
    %  ---------------PRESENCIA DE HUECOS--------------
    % -------------------------------------------------
    
    % Binarización de la imagen, buscar componentes conectados en imagen binaria
    % y aplicación del algoritmo Excentricity
    te = strel('square',2);
    K = imclose(im4b3n,te);
    C= imfill(im4b3n,'holes');
    
    Circulo= C .* K;
    Circulo1 = imabsdiff(double(C), double(Circulo));
    
    cc3 = bwconncomp(Circulo1);
    stat = regionprops(cc3,'Area');
    L = length(stat);
    for l=1:L
        if stat(l).Area > 5
            hueco=1;
        else
            hueco=0;
        end
    end
    
    % -------------------------------------------------
    %  -----NÚMERO DE PUNTOS FINALES DEL ESQUELETO-----
    % -------------------------------------------------
    
    % Binarización de la imagen y aplicación de filtro
    im4r_binarized = not(imbinarize(B));
    
    % Aplicacion de operaciones morfologicas
    im4r_binarized2 = imerode(im4r_binarized,1);
    BW3 = bwmorph(im4r_binarized2,'thin',Inf);
    terminating_pts = find_skel_ends(BW3);
    puntos_finales = length(terminating_pts);
    
    % -------------------------------------------------
    %  -------------DIFERENCIA CONVEXA-----------------
    % ------------------------------------------------- 
    
    % Conjunto convexo
    CH = bwconvhull(im4b3n);
    
    % Obtencion de las convexidades
    im_convex = CH - im4b3n;
    
    cc2 = bwconncomp(im_convex);
    areas = regionprops(cc2, 'Area', 'Eccentricity');
    L = length(areas);
    convexidad = 0;
    
    % Comprobar si es convexa o no
    for o=1:L
        if areas(o).Area > 40
            convexidad = convexidad + 1;
        end
    end
    
    % -------------------------------------------------
    %  --------EXCENTRICIDAD DE LA IMAGEN--------------
    % -------------------------------------------------
    
    % Se trata de determinar el grado de desviación de una sección cónica
    % Comprender el valor entre 0 y 1
    
    % Binarización de la imagen, buscar componentes conectados en imagen binaria
    % y aplicación del algoritmo Excentricity
    C= imfill(im4b3n,'holes');
    cc = bwconncomp(C);
    stat = regionprops(cc,'Area', 'Eccentricity');
    imgFeatures= stat.Eccentricity;
    
    % Añadimos la excentricidad de la imagen a sum
    suma=0;
    suma=suma+imgFeatures;
    if suma > 0.50
        excentricidad = 1;
    else
        excentricidad = 0;
    end
    
% VARIABLES QUE ALMACENAN RESULTADOS DE LOS DESCRIPTORES

%  ----------ORIENTACIÓN DE LA IMAGEN--------------
%         S = 1; % 1 -> VERTICAL
%         S = 0; % 0 -> HORIZONTAL

%  ---------------GRUPOS DE DEDOS------------------
%         dedo; NUMERO DE GRUPOS DE DEDOS

%  ----DISCONTINUIDADES EN LA REGIÓN CENTRAL-------
%         des=sum(sum(Aint));

%  ---------------PRESENCIA DE HUECOS--------------
%         hueco=1; HUECOS SI
%         hueco=0; HUECOS NO

%  -----NÚMERO DE PUNTOS FINALES DEL ESQUELETO-----
%         puntos_finales = length(terminating_pts); NÚMERO DE PUNTOS FINALES

%  -------------DIFERENCIA CONVEXA-----------------
%           NUMERO DE convexidades

%  --------EXCENTRICIDAD DE LA IMAGEN--------------
%     if suma > 0.50
%         excentricidad = 1;
%     else
%         excentricidad = 0;
%     end

X(i,:) = [S dedo des hueco puntos_finales convexidad excentricidad];  % Descriptores
end
end