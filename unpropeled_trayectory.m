function [Rbm, trayectoria_real, velocidad_real] = unpropeled_trayectory(blancos, i, t)
%% Con esta trayectoria se está suponiendo que el blanco no tiene o ha perdido su propulsión. 
% Con esta trayectoria se está suponiendo que el blanco no tiene o ha perdido su propulsión.
% Las velocidades y las distancias iniciales no son las distancias desde las que se lanza el proyectil,
% sino las primeras en rango.



% No está prgramado para más de un blanco de momento

    

plot (t, Rbm)
% https://www.grc.nasa.gov/www/k-12/rocket/termvr.html
% https://science.howstuffworks.com/rpg3.htm 
% https://en.wikipedia.org/wiki/Rheinmetall_Rh-120 120mm projectile

%http://hyperphysics.phy-astr.gsu.edu/hbasees/avari.html movimiento pòli