function [historial] = historial_aux (historial, promoEstablecida, trazas_muertas,maxCal, config)
    % Función para manejar la limpieza del historial
    switch config.kalman_type
        case 0
            for numDead = trazas_muertas
               if ~(nnz(historial.trazas.r(:,numDead)) < maxCal) % Todas las trazas que hayan estado vivas durante maxCal
                  historial.descolgadas.trazas.r = cat(2, historial.descolgadas.trazas.r, historial.trazas.r(:,numDead));
                  historial.descolgadas.trazas.v = cat(2, historial.descolgadas.trazas.v, historial.trazas.v(:,numDead));

                  historial.descolgadas.raw.r = cat(2, historial.descolgadas.raw.r, historial.raw.r(:,numDead));
                  historial.descolgadas.raw.v = cat(2, historial.descolgadas.raw.v, historial.raw.v(:,numDead));
               end
            end
            historial.trazas.r(:,trazas_muertas) = []; 
            historial.trazas.v(:,trazas_muertas) = []; 

            historial.raw.r(:,trazas_muertas) = []; 
            historial.raw.v(:,trazas_muertas) = [];
        case 1
            for numDead = trazas_muertas
               if ~(nnz(historial.trazas.r(:,numDead)) < maxCal) % Todas las trazas que hayan estado vivas durante maxCal
                  historial.descolgadas.trazas.r = cat(2, historial.descolgadas.trazas.r, historial.trazas.r(:,numDead));
                  historial.descolgadas.trazas.v = cat(2, historial.descolgadas.trazas.v, historial.trazas.v(:,numDead));
                  historial.descolgadas.trazas.a = cat(2, historial.descolgadas.trazas.a, historial.trazas.a(:,numDead));

                  historial.descolgadas.raw.r = cat(2, historial.descolgadas.raw.r, historial.raw.r(:,numDead));
                  historial.descolgadas.raw.v = cat(2, historial.descolgadas.raw.v, historial.raw.v(:,numDead));
                  historial.descolgadas.raw.a = cat(2, historial.descolgadas.raw.a, historial.raw.a(:,numDead));

               end
            end
            historial.trazas.r(:,trazas_muertas) = []; 
            historial.trazas.v(:,trazas_muertas) = []; 
            historial.trazas.a(:,trazas_muertas) = [];

            historial.raw.r(:,trazas_muertas) = []; 
            historial.raw.v(:,trazas_muertas) = [];
            historial.raw.a(:,trazas_muertas) = [];
    end