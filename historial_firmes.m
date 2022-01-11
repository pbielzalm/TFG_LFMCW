function [tracked_data, raw_data] = historial_firmes (historial, config, maxCal)
   %% Función para juntar todo el historial de las trazas que en algún momento han sido firmes
    % Borrar las trazas que no hayan llegado a maxcal
    tentativas_index = historial.trazas.r(config.num_bursts-maxCal,:) == 0;
    
    historial.trazas.r(:,tentativas_index) = [];
    historial.trazas.v(:,tentativas_index) = [];
    
    historial.raw.r(:,tentativas_index) = [];
    historial.raw.v(:,tentativas_index) = [];
    
    %Borramos la columna de 0s auxiliar
    empty_indexes = ~any(historial.descolgadas.trazas.r,1);
    
    historial.descolgadas.trazas.r( :, empty_indexes ) = [];  %columns
    historial.descolgadas.trazas.v( :, empty_indexes ) = [];  %columns
    
    historial.descolgadas.raw.r( :, empty_indexes ) = [];  %columns
    historial.descolgadas.raw.v( :, empty_indexes ) = [];  %columns
    
    tracked_data.r = cat(2, historial.trazas.r, historial.descolgadas.trazas.r);
    tracked_data.v = cat(2, historial.trazas.v, historial.descolgadas.trazas.v);
    
    raw_data.r = cat(2, historial.raw.r, historial.descolgadas.raw.r);
    raw_data.v = cat(2, historial.raw.v, historial.descolgadas.raw.v);
    
    switch config.kalman_type
        case 1
            historial.trazas.a(:,tentativas_index) = [];
            historial.raw.a(:,tentativas_index) = [];
            historial.descolgadas.trazas.a( :, empty_indexes ) = [];  %columns
            historial.descolgadas.raw.a( :, empty_indexes ) = [];  %columns
            tracked_data.a = cat(2, historial.trazas.a, historial.descolgadas.trazas.a);
            raw_data.a = cat(2, historial.raw.a, historial.descolgadas.raw.a);
    end
%     % Guardadas al revés
%     tracked_data.r = cat(2, historial.descolgadas.trazas.r,historial.trazas.r);
%     tracked_data.v = cat(2, historial.descolgadas.trazas.v, historial.trazas.v);
%     
%     raw_data.r = cat(2, historial.descolgadas.raw.r, historial.raw.r);
%     raw_data.v = cat(2, historial.descolgadas.raw.v, historial.raw.v);
    
    
