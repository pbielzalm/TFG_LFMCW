function [est_acceleration] = acceleration_est (interpoledPlot, historial, delta_t, nburst,system)
    %Inicializamos los parámetros
    [~,longPlots]=size(historial.raw.r); %Longitud de los plots antiguos
    est_acceleration = zeros(1, longPlots);
    dth=1000; % metros de diferencia
    
    for i = length(interpoledPlot.r)
    %Comprobamos plot a plot si correla con alguno anterior 
        distMin=inf; %Distancia mínima para ver a qué plot antiguo es la que más se acerca
        posMin=0; %posición del plot a la que más se acerca
        for j=1:longPlots

            difObs = abs((interpoledPlot.r(i) - historial.raw.r(nburst-1,j)));
            
            
            if(difObs<distMin) %Para comprobar mínima 
                distMin=difObs; 
                posMin=j;
            end
        end
        
        if(distMin<dth) %Si baja el umbral, se coloca en la traza que habíamos visto 
            est_acceleration(i)=(interpoledPlot.v(i)-historial.raw.v(nburst-1,posMin))/delta_t;
        else
            est_acceleration(i) = 0;
        end
    end

