    function [trazas, conjuntoCovarianzas] = eliminaTrazas(trazas, condicion, conjuntoCovarianzas, config)
    trazas.r(condicion)=[];
    trazas.v(condicion)=[];
    trazas.estado(condicion)=[];
    trazas.calidad(condicion)=[];
    trazas.extrapol_counter(condicion)=[];
    trazas.no_corr(condicion)=[];
    conjuntoCovarianzas(condicion) = [];
%     detections_to_track(condicion) = [];
switch config.kalman_type
        
    case 1
        trazas.a(condicion)=[];
end