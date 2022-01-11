function [detecciones] = deteccion_CFAR(Mrd, eje_R , eje_vel, procesador)

%Crear estructura detecciones (R y V), guardar potencia
%********************************************************************
% Realizar CA-CFAR sobre la matriz Distancia-Doppler
%**********************************************************************

guardWindow = procesador.Mg; %longitud de la ventana de guarda (incluido el CUT). 
sizeTraining = procesador.M_ref;
trainingWindow = sizeTraining + guardWindow; %número de celdas de entrenamiento a cada lado de la ventana de guarda más ventana de guarda
[m, n] = size(Mrd);



tA = matlab.tall.movingWindow(@sum,[trainingWindow,trainingWindow],Mrd);
tG = matlab.tall.movingWindow(@sum,[guardWindow,guardWindow],Mrd);
threshold = zeros(size(Mrd));
detecciones.matrix = zeros(size(Mrd));
o = 0;

for i = 1:numel(tA)
    nCol = 2*trainingWindow+1;
    nCol_g = 2*guardWindow+1;
    nRow = nCol;
    nRow_g = nCol_g;
    col = ceil(i/m);
    row = i - (col-1)*m;
    tAux = trainingWindow;
    tAux_g = guardWindow;
    leftWindow = tAux;
    rightWindow = tAux;
    leftWindow_g = tAux_g;
    rightWindow_g = tAux_g;
    nAux = n;
    nAux_g = n;
    k = zeros(1,2*tAux);
    g = zeros(1,2*tAux_g);
    for j = 1:trainingWindow
        %w_total
        if col-tAux <= 0     
          leftWindow = leftWindow-1;
          k(j) = tA(i+m*j);
          tAux = tAux - 1;    
        elseif col+tAux > nAux
          rightWindow = rightWindow-1;
          nAux = nAux+1;
          k(j) = tA(i-m*j);
        else
        k(j) = tA(i+m*rightWindow) + tA(i-leftWindow*m);
        leftWindow = leftWindow-1;
        rightWindow = rightWindow-1;
        end   
    end
    for l = 1:guardWindow
     %W_guarda 
            if col-tAux_g <= 0    
              leftWindow_g = leftWindow_g-1;
              g(l) = tG(i+m*l);
              tAux_g = tAux_g - 1;    
            elseif col+tAux_g > nAux_g
              rightWindow_g = rightWindow_g-1;
              nAux_g = nAux_g+1;
              g(l) = tG(i-m*l);
            else 
            g(l) = tG(i+m*rightWindow_g) + tG(i-m*leftWindow_g);
            leftWindow_g = leftWindow_g-1;
            rightWindow_g = rightWindow_g-1;
            end
    end
    %número de training cells (cellstotal-cellsguard)
    if (col-trainingWindow <= 0) || (col+trainingWindow >= n)
        A = [trainingWindow-(n-col), trainingWindow-(col-1)];
        nCol = nCol - max(A);
    end
    if (col-guardWindow <= 0) || (col+guardWindow >= n)
        A_g = [guardWindow-(n-col), guardWindow-(col-1)];
        nCol_g = nCol_g - max(A_g);
    end
    
    if (row-trainingWindow <= 0) || (row+trainingWindow >= m)
        A = [trainingWindow-(m-row), trainingWindow-(row-1)];
        nRow = nRow - max(A);
    end
    if (row-guardWindow <= 0) || (row+guardWindow >= m)
        A_g = [guardWindow-(m-row), guardWindow-(row-1)];
        nRow_g = nRow_g - max(A_g);
    end
    trainingElements = (nCol*nRow)-(nCol_g*nRow_g);
    thresholdFactor = trainingElements*(power(procesador.pfa,-1/trainingElements)-1);
    threshold(i) = 10*log10(thresholdFactor*rdivide((sum(k)+tA(i)) - (sum(g)+tG(i)),trainingElements).^2);
    Mrd_dB = 20 * log10(Mrd(i));
    if(threshold(i) >= Mrd_dB)
      detecciones.matrix(i) = 0;
    else
      detecciones.matrix(i) = Mrd_dB;  
      o = o+1;  
      detecciones.Power_dB(o) = Mrd_dB;
      detecciones.Velocity(o) = eje_vel(row);
      detecciones.Range(o) = eje_R(col);
    end
end
    
%   figure
%   Mrd2 = 20*log10(Mrd);
%   plot(threshold(59,:));
%   hold on
%   plot(Mrd2(59,:));
%   plot(Mrd2(7,:));
%   plot(threshold2(7,:));
%   hold off
%   detecciones.matrix(59,462:490)

% figure
% imagesc(eje_R, eje_vel, detecciones.matrix); % 
% colormap(jet);
% colorbar;
