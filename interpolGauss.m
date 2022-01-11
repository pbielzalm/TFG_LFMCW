function [interpoledPlot] = interpolGauss(plotDet)
    [numBlancos, tamBlob_max] = size(plotDet.pot);
    interpoledPlot.r = zeros(1,numBlancos);
    interpoledPlot.v = zeros(1,numBlancos);
    interpoledPlot.pot = zeros(1,numBlancos);
    
for i=1:numBlancos
        %find maxPot_index
        potPlot = plotDet.pot(i,:);
        [maxPot,maxPot_index]=max(potPlot);
        rangePlot = plotDet.r(i,:);
        [debug,~]= size(plotDet.v);
        if (i > debug)
            disp('warning');
        end
        velocityPlot = plotDet.v(i,:);
        
        sk = maxPot;        
        k_m_V = velocityPlot(maxPot_index);
        k_m_R = rangePlot(maxPot_index);
        
        if maxPot_index == 1 || maxPot_index == tamBlob_max
            interpoledPlot.pot(i) = maxPot;
            interpoledPlot.r(i) = k_m_R;
            interpoledPlot.v(i) = k_m_V;
        else
            
            %Buscamos los valores para interpolar en velocidad
            if rangePlot(maxPot_index-1) == rangePlot(maxPot_index+1)
                %si los valores anterior y siguiente en la matriz range son iguales, significa que están en la misma columna
                k_next_V = maxPot_index+1;  
                k_prev_V = maxPot_index-1;
                %interpolar en V
                %Hay que tener en cuenta que el orden de las frecuencias
                %a estudiar importan para el signo (logaritmo del numerador)
                if abs(velocityPlot(k_next_V)) > abs(velocityPlot(k_prev_V))
                    sk_next_V = potPlot(k_next_V);
                    sk_prev_V = potPlot(k_prev_V);
                else
                    sk_prev_V = potPlot(k_next_V);
                    sk_next_V = potPlot(k_prev_V);
                end
                sign = k_m_V/abs(k_m_V); 
                interpoledPlot.v(i) = sign*(sign*k_m_V + (log(sk_next_V/sk_prev_V)/(2*log(sk^2/(sk_prev_V*sk_next_V)))));
            else
                interpoledPlot.v(i)=k_m_V;
            end
            %Buscamos los valores para interpolar en distancia
            %No se puede hacer como en velocidad porque los valores están
            %metidos en la matriz de arriba a abajo, no de izquierda a
            %derecha
            
            
            k_next_R_aux = find(rangePlot<rangePlot(maxPot_index));
            k_prev_R_aux = fliplr(find(rangePlot>rangePlot(maxPot_index))); %flipeamos el vector para recorrerlo primero
                                                                            %desde los valores más cercanos a k_m
            
            if ~isempty(k_next_R_aux) && ~isempty(k_prev_R_aux)
                %Calculamos k_next_r
                k_next_R = [];
                [~,nextLength]=size(k_next_R_aux);
                for j=1:nextLength
                    if velocityPlot(k_next_R_aux(j)) == velocityPlot(maxPot_index)
                        k_next_R = k_next_R_aux(j);
                        break
                    end
                end
                %Calculamos k_prev_R
                k_prev_R = [];
                [~,prevLength]=size(k_prev_R_aux);
          
                for j=1:prevLength
                    if velocityPlot(k_prev_R_aux(j)) == velocityPlot(maxPot_index)
                        k_prev_R = k_prev_R_aux(j);
                        break
                    end
                end
                if  isempty(k_next_R) || isempty(k_prev_R)
                    interpoledPlot.r(i)=k_m_R;
                else    
                    %interpolar en R
                    %sk = maxPot;
                    %Hay que tener en cuenta que el orden de las frecuencias
                    %a estudiar importan para el signo (logaritmo del
                    %numerador)
                    if abs(rangePlot(k_next_R)) > abs(rangePlot(k_prev_R))
                        sk_next_R = potPlot(k_next_R);
                        sk_prev_R = potPlot(k_prev_R);
                    else
                        sk_prev_R = potPlot(k_next_R);
                        sk_next_R = potPlot(k_prev_R);
                    end
                    sign = k_m_R/abs(k_m_R); 
                    interpoledPlot.r(i) = sign*(sign*k_m_R + (log(sk_next_R/sk_prev_R)/(2*log(sk^2/(sk_prev_R*sk_next_R)))));
                end
            else
                interpoledPlot.r(i)=k_m_R;
            end
        end
end