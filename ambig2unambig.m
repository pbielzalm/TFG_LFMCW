function [unambig_velocity] = ambig2unambig (trazas, senal)
        unambig_velocity = zeros(1,length(trazas.v));
        max_vel = 2*senal.lambda/(4*senal.PRI); % por 2 porque es el rango de velocidades
        v_aux_inter = floor(abs(2*trazas.v)./(max_vel)); % >1 si es ambigua
        v_aux = (2*abs(trazas.v)/max_vel) - v_aux_inter;
        signo = trazas.v./abs(trazas.v);
        for j = 1:length(trazas.v)
            if (v_aux(j) < 0.5) && (v_aux_inter(j)>0)   % Si es ambigua y negativa
               unambig_velocity(j) = -(((max_vel/2)*(1-v_aux(j))))*signo(j);
            elseif (v_aux(j) >= 0.5) && (v_aux_inter(j)>0) % Si es ambigua y positiva
                unambig_velocity(j) = (((max_vel/2)*(1-v_aux(j))))*signo(j);
            elseif v_aux(j) < 0.5
                unambig_velocity(j) = signo(j)*((max_vel/2) * v_aux(j));
            else
                unambig_velocity(j) = signo(j)*((max_vel/2) * v_aux(j));
            end
        end
        
%         if unambig_velocity ~= trazas.v
%             disp("unambig_velocity fail");
%         end