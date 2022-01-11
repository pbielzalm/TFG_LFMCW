clear
clc
close all

%% Parametros script
config.verbose = 0; %sacar figuras
config.kalman_type =0; % NCV = 0; NCA = 1
config.target_type = 1; % MRU = 0; MRUA = 1 (RPG-7); Tiro_tenso = 2; Movimiento con aceleración variable = 3;
config.modo = 5; % Modo menú = -1; RPG-7 = 7 (suponemos MRUA); Artillería = 1 (se fuerza tiro tenso) 
                 % Un proyectil que adelanta a otro = 2; Blanco maniobrante = 3; 
                 % Loitering Drone = 4; ; Lotering + RPG-7 = 5
%% Configuracion de la simulación
config.num_bursts = 100; %Número de burst grabados
config.burst_rep_period = 20e-03; %Periodo de repetición entre bursts en (s)
config.max_detections_expected = 12;
%% Parametros señal transmitida simulada
senal.BW = 10e06; %Hz
senal.tau = 88e-06; %s. Duración rampa % 10 micro
senal.c = 299792458; %m/s
senal.f0 = 3.5e09; % Hz
senal.PRI = senal.tau; %s. Duración intervalo de tx/rx
senal.npulsos = 64; %numero de rampas por burst 
senal.lambda = senal.c/senal.f0; %m
senal.pendiente = +1; %Pendiente de la chirp +1 o -1
senal.gamma = senal.pendiente*senal.BW/senal.tau; % Hz/s

%% Datos auxiliares para simulacion
system.PndBq = 16; % Potencia de ruido en dBq
system.fs = senal.BW*1.2; %Hz Frecuencia de muestreo
system.nbits = 24; %Muestras IQ de 24 bits

% system.dist_res = senal.c*senal.PRI/2; % mal
%% Inicialización plot
plotDet.r=[];
plotDet.v=[];
plotDet.pot=[];


%% Parámetros trazas 
numExtrapConf=3;  %Nº máximo de extrapolaciones permitidas en Trazas Firmes
numExtrapTent=2; %Nº máximo de extrapolaciones permitidas en Trazas Tentativas
iniCalidad=3; %Nº máximo de extrapolaciones permitidas en Trazas Tentativas
maxCal=10; %Valor de calidad máxima
promoEstablecida=5; %Valor de calidad para generar Traza Firme
umbralDrop=3; %Valor de calidad mínima (umbral de drop
actCalcorr=1; %Actualización de la calidad de Trazas
actCalext=1; %Actualización de la calidad de Trazas
%% Parametros blanco simulado
%Inicialización de los parámetros del blanco. Para cambiar valores por
%defecto, mirar caso 3.
blancos.SWER = []; % tipo swerling 1, 3 o 5
blancos.SNR_ref=[]; %dB
blancos.R_ref = []; % (m) distancia a la que se obtiene la blancos.SNR_ref para el blanco
blancos.vrb = []; %(m/s) velocidad radial del blanco: Positiva se aleja, negativa se acerca
blancos.Rbini = []; %metros. Distancia inicial del blanco (m)
blancos.fdb = []; %Hz fdoppler blanco
blancos.acc = []; % Aceleración del blanco (m/s^2)

if (config.modo == -1)
    dlgTitle    = 'Elección de modo';
    dlgQuestion = 'Elija el modo de generación de blancos';
    blancos.modo = questdlg(dlgQuestion,dlgTitle,'Manual','Automático','Por defecto', 'Por defecto');
else
    blancos.modo = config.modo;
end

switch blancos.modo
    case 'Manual'
        disp("Has elegido el modo manual")
        blancos.N=inputdlg("Introduzca número de blancos","Número de blancos");
        blancos.N=str2num(blancos.N{1});
        for i=1:blancos.N
            prompt = {"Introduzca Swerling del blanco nº" + i,"Introduzca SNR de referencia del blanco nº" + i , "Introduzca distancia de referencia del blanco nº" + i, "Introduzca velocidad radial del blanco nº" + i ,"Introduzca distancia incial del blanco nº" + i};
            dlgtitle = "Blanco nº" + i;
            answer = inputdlg(prompt,dlgtitle);
            
            SWERaux=str2double(answer{1});
            SNR_refaux=str2double(answer{2})-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos);
            R_ref=str2double(answer{3});
            vrbaux=str2double(answer{4});
            rbiniaux=str2double(answer{5});
            fdbaux=-2*vrbaux/senal.lambda;
            
            blancos.SWER =[blancos.SWER SWERaux ];
            blancos.SNR_ref=[blancos.SNR_ref SNR_refaux];
            blancos.R_ref =[blancos.R_ref R_ref] ;
            blancos.vrb = [blancos.vrb vrbaux ];
            blancos.Rbini = [blancos.Rbini rbiniaux ];
            blancos.fdb = [blancos.fdb fdbaux];
            
        end
        
    case 'Automático'
        disp('Has elegido el modo automático')
        blancos.N=inputdlg("Introduzca número de blancos","Número de blancos");
        blancos.N=str2num(blancos.N{1});
        
        valuesSwerling=[1,3,5];
        SNRrefmin=10; SNRrefmax=20;
        Rrefmin=1000;   Rrefmax=4000;
        vrbmin=-(1/senal.PRI)*senal.lambda/4;    vrbmax=(1/senal.PRI)*senal.lambda/4;
        rbinimin=0;   rbinimax=(senal.c*senal.PRI/2)*0.1;
        for i=1:blancos.N
            SWERaux=valuesSwerling(randi(length(valuesSwerling)));
            SNR_refaux=(SNRrefmax-SNRrefmin)*rand + SNRrefmin -10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); 
            R_ref=(Rrefmax-Rrefmin)*rand + Rrefmin;
            vrbaux=(vrbmax-vrbmin)*rand + vrbmin;
            rbiniaux=(rbinimax-rbinimin)*rand + rbinimin;
            fdbaux=-2*vrbaux/senal.lambda;
            
            blancos.SWER =[blancos.SWER SWERaux ];
            blancos.SNR_ref=[blancos.SNR_ref SNR_refaux];
            blancos.R_ref =[blancos.R_ref R_ref] ;
            blancos.vrb = [blancos.vrb vrbaux ];
            blancos.Rbini = [blancos.Rbini rbiniaux ];
            blancos.fdb = [blancos.fdb fdbaux];
        end
        
    case 'Por defecto'
        disp("Has elegido el modo por defecto")
        %Introducir aquí valores por defecto del código
        blancos.SWER = 1; 
        blancos.SNR_ref=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); 
        %blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1; 
        blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1;
        blancos.vrb = -150; 
        blancos.Rbini = 1000; 
        blancos.fdb = -2*blancos.vrb/senal.lambda; 
        blancos.acc = 100;
        
        blancos.N=length(blancos.Rbini);
        
    case 7
        disp("RPG-7")
        %Introducir aquí valores por defecto del código
        config.target_type = 1; % Forzamos MRUA
        blancos.SWER = 1;
        blancos.SNR_ref=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); 
        blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1*3; 
        blancos.vrb = -290; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini = 2000; 
        blancos.fdb = -2*blancos.vrb/senal.lambda;
        blancos.acc = -126; % aceleración inicial
        
        blancos.N=1;
        
    case 1
        disp("Artillería")
        %Introducir aquí valores por defecto del código
        config.target_type = 2; % Forzamos tiro tenso
        blancos.SWER = 1; 
        blancos.SNR_ref=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); % ??? cambiar???
        blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1*4; 
        blancos.vrb = -1470 ;%-1670; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini = 4000; 
        blancos.fdb = -2*blancos.vrb/senal.lambda;
        
        blancos.m = 18.6; % mass of the projectile (M829 projectile)
        
        blancos.C = 0.5 * 1.29 * 0.7 * 0.25;%0.5 * densidad del aire * coeficiente de rozamiento * superficie del proyectil (M829 projectile)
        
        
        blancos.N=1;
        
    case 2
        disp("Artellería a distintas velocidades") % no es interesante que ya que uno solo no lo sigue bien
        %Introducir aquí valores por defecto del código
        config.target_type = 2; % Forzamos tiro tenso
        
        blancos.SWER(1) = 1; 
 
        blancos.SNR_ref(1)=0; % ??? cambiar???
        blancos.R_ref(1) = 2*(senal.c*senal.PRI/2)*0.1; 
        blancos.vrb(1) = -1670; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini(1) = 4000; 
        blancos.fdb(1) = -2*blancos.vrb(1)/senal.lambda;
        
        blancos.m(1) = 4.27; % mass of the projectile (M829 projectile)
        blancos.C(1) = 0.5 * 1.29 * 0.7 * 0.25;%0.5 * densidad del aire * coeficiente de rozamiento * superficie del proyectil (M829 projectile)
        
        blancos.SWER(2) = blancos.SWER(1);
        blancos.SNR_ref(2)=blancos.SNR_ref(1); % ??? cambiar???
        blancos.R_ref(2) = 2*(senal.c*senal.PRI/2)*0.1; 
        blancos.vrb(2) = -600; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini(2) = 3000; 
        blancos.fdb(2) = -2*blancos.vrb(2)/senal.lambda;
        
        blancos.m(2) = 4.27; % mass of the projectile (M829 projectile)
        blancos.C(2) = 0.5 * 1.29 * 0.7 * 0.25;%0.5 * densidad del aire * coeficiente de rozamiento * superficie del proyectil (M829 projectile)
        
        
        blancos.N=2;
        
    case 3
        disp("Blanco maniobrante")
        %Introducir aquí valores por defecto del código
        config.target_type = 3; % Forzamos aceleración variable
        blancos.SWER = 1; 
        blancos.SNR_ref=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); % ??? cambiar???
        blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1; 
        blancos.vrb = 300; % velocidad inicial radial
        blancos.Rbini = -1000; 
        blancos.fdb = -2*blancos.vrb/senal.lambda; 
        blancos.acc = 20; % aceleración inicial
        
        loiter.c = blancos.acc*2; % el nombre de la estructura no es correcto ya que no es para loitering
        loiter.d = blancos.acc*-4.5;
        
        blancos.N=1;
        
    case 4
        disp("Loitering Drone")
        %Introducir aquí valores por defecto del código
        config.target_type = 1; % Forzamos MRUA
        blancos.SWER = 1; 
        blancos.SNR_ref=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos); % ??? cambiar???
        blancos.R_ref = 2*(senal.c*senal.PRI/2)*0.1*0.3; 
        blancos.vrb = 0; % velocidad inicial radial
        blancos.Rbini = 270; 
        blancos.fdb = -2*blancos.vrb/senal.lambda; 
        blancos.acc = -42.5; % aceleración inicial % unos 32.5 + gravedad
        
        
        blancos.N=1;
        
    case 5
        disp("Loitering + RPG") % no es interesante que ya que uno solo no lo sigue bien
        %Introducir aquí valores por defecto del código
        config.target_type = 1; % Forzamos MRUA
        %Loit
        blancos.SWER(1) = 1; 
 
        blancos.SNR_ref(1)=15.3-10*log10(system.fs*senal.tau)-10*log10(senal.npulsos);
        blancos.R_ref(1) = 2*(senal.c*senal.PRI/2)*0.1*0.3; 
        blancos.vrb(1) = 0; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini(1) = 300; 
        blancos.fdb(1) = -2*blancos.vrb(1)/senal.lambda;
        blancos.acc(1) = -42.5; % aceleración inicial
        
        %RPG-7
        blancos.SWER(2) = blancos.SWER(1);
        blancos.SNR_ref(2)=blancos.SNR_ref(1); 
        blancos.R_ref(2) = 2*(senal.c*senal.PRI/2)*0.1; 
        blancos.vrb(2) = -290; % velocidad inicial radial (viene prácticamente de frente --> == velocidad inicial)
        blancos.Rbini(2) = 900; 
        blancos.fdb(2) = -2*blancos.vrb(2)/senal.lambda;
        blancos.acc(2) = -126; % aceleración inicial
        
        blancos.N=2;
    otherwise
        disp('Introduzca un modo de funcionamiento válido')
end

 dlgTitle    = 'Visualización de información de blancos';
 dlgQuestion = '¿Desea visualizar información sobre los blancos generados en la consola?';
 choice = questdlg(dlgQuestion,dlgTitle,'Sí','No','Sí');
 switch choice
     case 'Sí'
         disp("  ")
         for i=1:blancos.N
             disp("Información sobre blanco nº" +i)
             disp("Swerling:" +blancos.SWER(i))
             disp("SNR de referencia: " + blancos.SNR_ref(i) + " dB")
             disp("Distancia referencia: " + blancos.R_ref(i) + " m")
             disp("Velocidad radial: " + blancos.vrb(i) + " m/s")
             disp("Distancia inicial del blanco: " + blancos.Rbini(i) + " m")
             disp("Frecuencia de Doppler del blanco: " + blancos.fdb(i) + " Hz")
             disp("  ");
         end
 end

%% Datos Procesado Range/Doppler (MTD)
procesador.M_ref = 2; %Dimensiones ventana cuadrada de referencia a derecha(e izquierda) arriba(y abajo) del CUT (CFAR)
procesador.Mg = 1; %Muestras de guarda del CFAR
procesador.ncoefMTD = senal.npulsos;
procesador.pfa = 1e-6; %probabilidad de falsa alarma CFAR
procesador.SLL_Range = -50; %dB
procesador.SLL_Doppler = -50; %dB

%% Eje de tiempos de señal recibida
num_samp_cpi = senal.PRI*system.fs*senal.npulsos;
t = [0:num_samp_cpi-1]*1/system.fs;

%% Parametros de precision radar
dev_tip_R = (senal.c/2/senal.BW)/sqrt(12); %m. Precision distancia. Asumiendo error uniforme
dev_tip_V = senal.lambda/2/(senal.npulsos*senal.PRI)/sqrt(12); %m/s. Precisión velocidad. Asumiendo error uniforme

%% Parametros iniciales Kalman
%sigma_R=dev_tip_R;
delta_t = senal.pendiente*senal.f0*senal.tau/senal.BW; %(senal.f0 - senal.BW/2)*(senal.tau)/senal.BW;
delta = config.burst_rep_period; %t(k)-t(k-1)
delta2 = delta^2;
delta3 = delta^3;
sigma_R=dev_tip_R;

switch config.kalman_type
    case 0
        %% Nearly Constant Velocity
        F = [1 delta; 0 1];
        H = [1 0];
        q =10e-04;
        GQGt = q * [delta3/3 delta2/2; delta2/2 delta]; % La primera delta estaba al cuadrado en vez de al cubo al hacer Montecarlo
        R=(sigma_R.^2)/.1;
        %P=[dev_tip_R^2  0; 0  (dev_tip_V^2)];
        P_ini=[(dev_tip_R^2)/10  0; 0  (dev_tip_V^2)*1000000]; % 10 y 1000000 funcionan 200 y 350, 1600 regular
        %P_ini =[dev_tip_R^2  0; 0  (dev_tip_V^2)]; % para no ambigu
        % P=[15 0; 0 50];
    
    case 1
        %% Nearly Constant Acceleration
        delta4 = delta^4;
        F = [1 delta delta2/2; 0 1 delta; 0 0 1];
        H = [1 0 0];
        accel_difference_max = (35.5 * delta_t);%m/s^2
        sigma_vk_square = (0.75*accel_difference_max)^2;
        GQGt = sigma_vk_square * [delta4/4 delta3/2 delta2/2; delta3/2 delta2 delta; delta/2 delta 1];
        
        R=(sigma_R.^2)/0.114210526315789;
        
        P_ini=[(dev_tip_R^2)/10  0 0; 0  (dev_tip_V^2)*1000000 0; 0 0 (dev_tip_V^2)*1000000];
        %P_ini =[dev_tip_R^2  0 0; 0  (dev_tip_V^2) 0;0 0 (dev_tip_V^2)*1000000 ];
end 
%% Bucle de CPIs (burst)
barra = waitbar(0,'Please wait...');
for nburst = 1:config.num_bursts
    
    barra = waitbar(nburst/config.num_bursts, barra, 'Generating data...');
    
    
    % Generamos ruido AWGN de potencia Pn   
    Pn = 10^(system.PndBq/10); %q^2
    srx = sqrt(Pn/2)*(randn(1,num_samp_cpi) +1i*randn(1,num_samp_cpi));
    
    
    %% *********************************************************************
    % EXTENDER A VARIOS BLANCOS CON UN BUCLE. LOS PARÁMETROS DE LOS BLANCOS
    % SERÁN VECTORES EN VEZ DE UN ÚNICO VALOR.
    %*********************************************************************
    
%     Rbm=zeros(blancos.N,length(t)); 
%     SNR=zeros(1,blancos.N); 
%     Pmed=zeros(1,blancos.N);
%     Pr=zeros(1,blancos.N);
%     db=zeros(blancos.N,length(t));
%     tp0=zeros(1,blancos.N);
%     pulsor=zeros(blancos.N,length(t));

    for i=1:length(blancos.Rbini)

        % Distancia actual del blanco al inicio del burst (m)
        switch config.target_type
            case 0  %MRU
                Rbm= blancos.Rbini(i) + blancos.vrb(i)*(nburst-1)*config.burst_rep_period;
                
                % SNR actual del blanco en el burst 
                SNR = blancos.SNR_ref(i)+40*log10(blancos.R_ref(i)/Rbm); %dB actualizados con la distancia
                
                % Calculamos distancia del blanco durante todo el burst  
                Rbm= Rbm + blancos.vrb(i)*t; %Se simula la migracion en distancia dentro del burst y de la propia rampa LFM
                db = 2*Rbm/senal.c; %delay ida y vuelta al blanco (s)
            case 1  % MRUA
                Rbm= blancos.Rbini(i) + blancos.vrb(i)*(nburst-1)*config.burst_rep_period + 0.5*blancos.acc(i)*((nburst-1)*config.burst_rep_period)^2;
                
                % SNR actual del blanco en el burst 
                SNR = blancos.SNR_ref(i)+40*log10(blancos.R_ref(i)/Rbm); %dB actualizados con la distancia
                
                % Calculamos distancia del blanco durante todo el burst  
                Rbm= Rbm + blancos.vrb(i)*t + 0.5*blancos.acc(i)*(t.^2); %Se simula la migracion en distancia dentro del burst y de la propia rampa LFM
                db = 2*Rbm/senal.c; %delay ida y vuelta al blanco (s)
            case 2 % Tiro tenso
                g = 9.8; % fuerza de la gravedad
                
                vt = blancos.m(i)*g/blancos.C(i); % velocidad terminal

                Rbm = blancos.Rbini(i) + (blancos.vrb(i)*vt/g)*(1-exp(-(g*(nburst-1)*config.burst_rep_period)/vt));
                
                % SNR actual del blanco en el burst 
                SNR = blancos.SNR_ref(i)+40*log10(blancos.R_ref(i)/Rbm); %dB actualizados con la distancia
                
                % Calculamos distancia del blanco durante todo el burst (v~cte) 
                Rbm= Rbm + blancos.vrb(i)*t; %Se simula la migracion en distancia dentro del burst y de la propia rampa LFM
                db = 2*Rbm/senal.c; %delay ida y vuelta al blanco (s)
            case 3 % Movimiento polinómico (blanco maniobrante)
                Rbm= blancos.Rbini(i) + blancos.vrb(i)*(nburst-1)*config.burst_rep_period + 0.5*blancos.acc(i)*(((nburst-1)*config.burst_rep_period)^2) + (1/6)*loiter.c(i)*(((nburst-1)*config.burst_rep_period)^3) + (1/12)*loiter.d(i)*(((nburst-1)*config.burst_rep_period)^4);
                SNR = blancos.SNR_ref(i)+40*log10(blancos.R_ref(i)/Rbm); %dB actualizados con la distancia
                % Calculamos distancia del blanco durante todo el burst (v~cte) 
                Rbm= Rbm + + blancos.vrb(i)*t + 0.5*blancos.acc(i)*(t.^2) + (1/6)*loiter.c(i)*(t.^3) + (1/12)*loiter.d(i)*(t.^4); %Se simula la migracion en distancia dentro del burst y de la propia rampa LFM
                db = 2*Rbm/senal.c; %delay ida y vuelta al blanco (s)
        end               

        % Calculamos potencia del blanco
        Pmed = 10^(SNR/10)*Pn; %W
        if blancos.SWER(i)==5
            Pr = Pmed; %W  ==>   SW5
        elseif blancos.SWER(i)==1
            Pr= -Pmed.*(log(rand(1,1)));   %exprnd(Pmed)); %SW1
        elseif blancos.SWER(i) ==3
            Pr = Pmed/4.*sum(randn(4,1).^2,1);  %chi2rnd(4,1,1).*Pmed/4; %SW3
        end

        

        if config.verbose
            figure(1);
            clf;
            figure(2);
            clf;
        end

        % Bucle de simulacion del burst
        for np = 1:senal.npulsos

            tp0 = (np-1)*senal.PRI;

            pulsor = sqrt(Pr).*rectpuls((t-(tp0+senal.tau/2+db))/senal.tau);
            pulsor= pulsor.*exp(-1i*2*pi*(tp0+senal.tau/2+db)*senal.f0).*exp(1i*pi*senal.gamma.*(t-(tp0+senal.tau/2+db)).^2);

            srx = srx + pulsor;

            if config.verbose
                figure(1);
                subplot(3,1,1);hold on;plot(t,real(pulsor));xlabel('Time, s');ylabel('I');title('Burst recibido sin ruido');
                subplot(3,1,2);hold on;plot(t,imag(pulsor));xlabel('Time, s');ylabel('Q');title('Burst recibido sin ruido');
                subplot(3,1,3);hold on;plot(t,abs(pulsor));xlabel('Time, s');ylabel('Envelope');title('Burst recibido sin ruido');

                figure(2);
                subplot(3,1,1);hold on;plot(t*senal.c/2,real(pulsor));xlabel('Range, m');ylabel('I');title('Burst recibido sin ruido');
                subplot(3,1,2);hold on;plot(t*senal.c/2,imag(pulsor));xlabel('Range, m');ylabel('Q');title('Burst recibido sin ruido');
                subplot(3,1,3);hold on;plot(t*senal.c/2,abs(pulsor));xlabel('Range, m');ylabel('Envelope');title('Burst recibido sin ruido');
            end
        end

         %% FIN DEL BUCLE DE SIMULACIÓN DE VARIOS BLANCOS % for 1:length(blancos.Rbini)
    end
     %*********************************************************************   
    
    
    %% Cuantificamos
    
    DataI = round(real(srx));
    DataI(DataI > 2^(system.nbits-1)-1) = 2^(system.nbits-1)-1;
    DataI(DataI < -2^(system.nbits-1)) = -2^(system.nbits-1);
    
    DataQ = round(imag(srx));
    DataQ(DataQ > 2^(system.nbits-1)-1) = 2^(system.nbits -1)-1;
    DataQ(DataQ < -2^(system.nbits-1)) = -2^(system.nbits-1);
    
    srx = DataI+1i*DataQ;
    clear DataI DataQ;
    
    if config.verbose    
        figure(3);
        subplot(3,1,1);hold on;plot(t,real(srx));xlabel('Time, s');ylabel('I');title('Burst recibido');
        subplot(3,1,2);hold on;plot(t,imag(srx));xlabel('Time, s');ylabel('Q');title('Burst recibido');
        subplot(3,1,3);hold on;plot(t,20*log10(abs(srx)));xlabel('Time, s');ylabel('Power, dB');title('Burst recibido');
        
        figure(4);
        subplot(3,1,1);hold on;plot(t*senal.c/2,real(srx));xlabel('Range, m');ylabel('I');title('Burst recibido');
        subplot(3,1,2);hold on;plot(t*senal.c/2,imag(srx));xlabel('Range, m');ylabel('Q');title('Burst recibido');
        subplot(3,1,3);hold on;plot(t*senal.c/2,20*log10(abs(srx)));xlabel('Range, m');ylabel('Power, dB');title('Burst recibido');
    end
    
    %%
    [Mrd, eje_R, eje_vel] = procesado_Range_Doppler_sucio(srx, t , senal, system, procesador);    
    
    [detecciones] = deteccion_CFAR(Mrd, eje_R , eje_vel, procesador);
    
    %% Crear plots con las detecciones pertenecientes a un mismo blanco.  
%Paso de detecciones a plots de manera dinámica
thresholdR=50;
thresholdV=50;
thresholdPot=20; %Si se quiere se puede utilizar para buscar picos de potencia en blancos seguidos. 

%Guardamos detecciones para comprobar
deteccionBackup=detecciones;

if isfield(detecciones,'Range')
    
    for i=1:length(detecciones.Range)
            cont=1;
            
            for j=1:length(detecciones.Range)
                if (detecciones.Range(j)==0) && (detecciones.Velocity(j)==0)
                    continue;
                elseif (abs(deteccionBackup.Range(i)-detecciones.Range(j))< thresholdR) && (abs(deteccionBackup.Velocity(i)-detecciones.Velocity(j))< thresholdV)
                    
                        plotDet.r(i,cont)=detecciones.Range(j);
                        plotDet.v(i,cont)=detecciones.Velocity(j);
                        plotDet.pot(i,cont)=detecciones.Power_dB(j);
                        % Eliminamos esas celdas porque ya las hemos
                        % incluido en un blanco
                        detecciones.Range(j)=0; %Usamos 0 para no volver a contar las detecciones
                        detecciones.Velocity(j)=0;
                        detecciones.Power_dB(j)=0;
                        cont=cont+1;
                else
                    continue;
                end
            end
            deteccionBackup.Range(i)=0; %Usamos 0 para no volver a contar las detecciones 
    end
%         [debug1,~]= size(plotDet.r);
%     [debug2,~]= size(plotDet.v);
%     if (debug1 ~= debug2)
%             disp('warning');
%     end
    

        %%
        %Para eliminar filas completas de 0s resultantes del algoritmo
        plotDet.r=plotDet.r(any(plotDet.v,2),any(plotDet.v,1));
        plotDet.v=plotDet.v(any(plotDet.v,2),any(plotDet.v,1));
        plotDet.pot=plotDet.pot(any(plotDet.v,2),any(plotDet.v,1));


        %Quitamos los 0s puntuales por Infinito para poder visualizarlos en los
        %plots
    %      plotDet.r(find(~plotDet.r))=Inf;
    %      plotDet.v(find(~plotDet.v))=Inf;
    %      plotDet.pot(find(~plotDet.pot))=Inf;
    %

    
        %Interpolación
        if(nburst > 1) && (config.kalman_type == 1)
            interpoledPlot_prev = interpoledPlot; %Guardamos el plot anterior para calcular la aceleración
        end

        [interpoledPlot] = interpolGauss(plotDet);



     if config.verbose
        %Visualización
        for i=1:length(plotDet.r(:,1))
            subplot(length(plotDet.r(:,1)),1,i);
            scatter(plotDet.r(i,:), plotDet.v(i,:),'filled');
            grid;
            axis([min(eje_R), max(eje_R), min(eje_vel), max(eje_vel)]);
    %         axis([-6.5892e+03,6.5892e+03,-235.7337, 243.3380]);
            title('Plot de blanco');
            xlabel('Rango (m)');
            ylabel('Velocidad (m/s)');
        end
     end



        switch config.kalman_type
            case 0 % NCV
                
                %% Asociar plots a trazas o en su defecto crear traza potencial.
                %Tipo de trazas: potencial (1), tentativa (2), firme (3), extrapolada
                %(4)
                                   
                if nburst==1
                    numeroPlots=length(interpoledPlot.r);
                    trazas=interpoledPlot;
                    detections_to_track = interpoledPlot;
                    trazas.calidad=iniCalidad*ones(1,numeroPlots); 
                    trazas.estado=ones(1,numeroPlots);
                    trazas.extrapol_counter=zeros(1,numeroPlots);
                    
                    conjuntoCovarianzas = cell(1,numeroPlots);
                    [conjuntoCovarianzas{:}] = deal(P_ini);
                    
                    historial.trazas.r = zeros(config.num_bursts, numeroPlots);
                    historial.trazas.v = zeros(config.num_bursts, numeroPlots);

                    historial.raw.r = zeros(config.num_bursts, numeroPlots);
                    historial.raw.v = zeros(config.num_bursts, numeroPlots);
                    
                    historial.trazas.r(1, :) = interpoledPlot.r;
                    historial.trazas.v(1, :) = interpoledPlot.v;
                    
                    historial.raw.r(1, :) = interpoledPlot.r;
                    historial.raw.v(1, :) = interpoledPlot.v;
                    
                    historial.descolgadas.trazas.r = zeros(config.num_bursts,1);
                    historial.descolgadas.trazas.v = zeros(config.num_bursts,1);
                    
                    historial.descolgadas.raw.r = zeros(config.num_bursts,1);
                    historial.descolgadas.raw.v = zeros(config.num_bursts,1);
                else
                    %Comprobamos las correlaciones
                    %Función que realiza la correlación de mahalanobis E INCLUYE EN
                    %TRAZAS LOS CAMBIOS DEBIDOS.

                    [detections_to_track, trazas, trazas_muertas, historial, promoEstablecida, conjuntoCovarianzas] = mahalanobis_NCV(interpoledPlot, trazasFiltradas, conjuntoCovarianzas, trazas, historial, nburst, config, delta, P_ini,senal);                    
                    %Eliminamos las trazas que no cumplen los requisitos
                    [trazas, conjuntoCovarianzas]=eliminaTrazas(trazas, trazas_muertas,conjuntoCovarianzas,config);
                    
                    [historial] = historial_aux (historial, promoEstablecida,trazas_muertas,maxCal, config);
                    
                    
                    
                    
                end
                %Reinicializamos valores para la siguiente vuelta 
                   
                plotDet=[];
                interpoledPlot=[];

               %% Kalman: Nearly Constant Velocity
               nDifTrazas = length(detections_to_track.r);
               trazasFiltradas = {nDifTrazas};
               for nTraza = 1:length(detections_to_track.r)
                    X= [trazas.r(nTraza) ;  trazas.v(nTraza)];
                    if detections_to_track.r(nTraza)~= 0     %inf si sustituimos los 0s por inf
                                    %posición detectada actual
                        Z = detections_to_track.r(nTraza);
                        P = cell2mat(conjuntoCovarianzas(nTraza));
                        [X,P] = kalman_NCV(F,GQGt,P,H,R,Z,X);
                    else
                         
                        
                          X=F*X;  % posición extrapolada si no se detecta % esto tiene sentido ?¿
                        
%                         Z=aux(1);
                        P = cell2mat(conjuntoCovarianzas(nTraza));
                        P = P*1;
                    end
                    
                    trazasFiltradas(nTraza) = {X};
                    conjuntoCovarianzas(nTraza) = {P};
                    trazas.r(nTraza) = X(1);
                    trazas.v(nTraza) = X(2);
               end
               
               
               
            case 1 
               %% NCA Kalman
                %% Asociar plots a trazas o en su defecto crear traza potencial.
                %Tipo de trazas: potencial (1), tentativa (2), firme (3), extrapolada
                %(4)
                %% Asociar plots a trazas o en su defecto crear traza potencial.
                %Tipo de trazas: potencial (1), tentativa (2), firme (3), extrapolada
                %(4)
                                   
                if nburst==1
                    numeroPlots=length(interpoledPlot.r);
                    trazas=interpoledPlot;
                    trazas.a = zeros(1, numeroPlots);
                    detections_to_track = interpoledPlot;
                    trazas.calidad=iniCalidad*ones(1,numeroPlots); 
                    trazas.estado=ones(1,numeroPlots);
                    trazas.extrapol_counter=zeros(1,numeroPlots);
                    
                    conjuntoCovarianzas = cell(1,numeroPlots);
                    [conjuntoCovarianzas{:}] = deal(P_ini);
                    
                    historial.trazas.r = zeros(config.num_bursts, numeroPlots);
                    historial.trazas.v = zeros(config.num_bursts, numeroPlots);
                    historial.trazas.a = zeros(config.num_bursts, numeroPlots);

                    historial.raw.r = zeros(config.num_bursts, numeroPlots);
                    historial.raw.v = zeros(config.num_bursts, numeroPlots);
                    historial.raw.a = zeros(config.num_bursts, numeroPlots);
                    
                    historial.trazas.r(1, :) = interpoledPlot.r;
                    historial.trazas.v(1, :) = interpoledPlot.v;
                    
                    historial.raw.r(1, :) = interpoledPlot.r;
                    historial.raw.v(1, :) = interpoledPlot.v;
                    
                    historial.descolgadas.trazas.r = zeros(config.num_bursts,1);
                    historial.descolgadas.trazas.v = zeros(config.num_bursts,1);
                    historial.descolgadas.trazas.a = zeros(config.num_bursts,1);
                    
                    historial.descolgadas.raw.r = zeros(config.num_bursts,1);
                    historial.descolgadas.raw.v = zeros(config.num_bursts,1);
                    historial.descolgadas.raw.a = zeros(config.num_bursts,1);
                else
                    %Comprobamos las correlaciones
                    %Función que realiza la correlación de mahalanobis E INCLUYE EN
                    %TRAZAS LOS CAMBIOS DEBIDOS.

                    [detections_to_track, trazas, trazas_muertas, historial, promoEstablecida, conjuntoCovarianzas] = mahalanobis_NCA(interpoledPlot, trazasFiltradas, conjuntoCovarianzas, trazas, historial, nburst, config, P_ini, delta_t, system, senal);                    
                    %Eliminamos las trazas que no cumplen los requisitos
                    [trazas, conjuntoCovarianzas]=eliminaTrazas(trazas, trazas_muertas,conjuntoCovarianzas,config);
                    
                    [historial] = historial_aux (historial, promoEstablecida,trazas_muertas,maxCal,config);
                    
                    
                    
                    
                end
                %Reinicializamos valores para la siguiente vuelta 
                   
                plotDet=[];
                interpoledPlot=[];

                %% Kalman: Nearly Constant Acceleration
               nDifTrazas = length(detections_to_track.r);
               trazasFiltradas = {nDifTrazas};
               for nTraza = 1:length(detections_to_track.r)
                    X= [trazas.r(nTraza) ;  trazas.v(nTraza); trazas.a(nTraza)];
                    if detections_to_track.r(nTraza)~= 0     %inf si sustituimos los 0s por inf
                                    %posición detectada actual
                        Z = detections_to_track.r(nTraza);
                        P = cell2mat(conjuntoCovarianzas(nTraza));
                        [X,P] = kalman_NCA(F,GQGt,P,H,R,Z,X);
                    else
                    
                        X=F*X;  % posición extrapolada si no se detecta % esto tiene sentido ?¿

%                         Z=aux(1);
                        P = cell2mat(conjuntoCovarianzas(nTraza));
                        P = P*1;
                    end
                    trazasFiltradas(nTraza) = {X};
                    conjuntoCovarianzas(nTraza) = {P};
                    trazas.r(nTraza) = X(1);
                    trazas.v(nTraza) = X(2);
                    trazas.a(nTraza) = X(3);
               end
        end
            disp(trazas.calidad);
    %       disp("Vuelta número "+ nburst);
    else
        disp('No detections');
end
end

elapsed_time = config.num_bursts*config.burst_rep_period;
elapsed_time_space = [0:(config.num_bursts-1)]*config.burst_rep_period;
[tracked_data,raw_data] = historial_firmes (historial,config, maxCal);

%likelihood_threshold_r = dev_tip_R*10;% Valor umbral de desviación del RMSE para ser asignado al blanco estudiado
likelihood_threshold_r = 200;
tracked_error.r = zeros(size(tracked_data.r));
detected_error.r = zeros(size(tracked_data.r));

tracked_error.v = zeros(size(tracked_data.r));
detected_error.v = zeros(size(tracked_data.r));

tracked_error.a = zeros(size(tracked_data.r));
detected_error.a = zeros(size(tracked_data.r));

for numBlanco= 1:blancos.N
    traces_index = [];
    RMSE_trace.r = [];
    RMSE_trace.v = [];
    RMSE_trace.a = [];
    RMSE_raw.r = [];
    RMSE_raw.v = [];
    RMSE_raw.a = [];
    switch config.target_type
        case 0
            trayectoria_real= blancos.Rbini(numBlanco) + blancos.vrb(numBlanco)*elapsed_time_space;
            velocidad_real = blancos.vrb(numBlanco)+zeros(1,config.num_bursts);
            aceleracion_real = 0*elapsed_time_space;
        case 1
            trayectoria_real= blancos.Rbini(numBlanco) + blancos.vrb(numBlanco)*elapsed_time_space + 0.5*blancos.acc(numBlanco)*(elapsed_time_space).^2;
            velocidad_real = blancos.vrb(numBlanco)+elapsed_time_space*blancos.acc(numBlanco);
            aceleracion_real= blancos.acc(numBlanco)+ 0*elapsed_time_space;
        case 2
            trayectoria_real= blancos.Rbini(numBlanco) + (blancos.vrb(numBlanco)*vt/g)*(1-exp(-(g*elapsed_time_space)/vt));
            velocidad_real = blancos.vrb(numBlanco)*exp((-elapsed_time_space*g)/vt);
            aceleracion_real = blancos.vrb(numBlanco)*exp((-elapsed_time_space*g)/vt);
        case 3
            trayectoria_real= blancos.Rbini(numBlanco) + blancos.vrb(numBlanco)*elapsed_time_space + 0.5*blancos.acc(numBlanco)*(elapsed_time_space.^2) - (1/6)*loiter.c(numBlanco)*(elapsed_time_space.^3) + (1/12)*loiter.d(numBlanco)*(elapsed_time_space.^4);
            velocidad_real= blancos.vrb(numBlanco) + blancos.acc(numBlanco)*(elapsed_time_space) + (1/2)*loiter.c(numBlanco)*(elapsed_time_space.^2) + (1/3)*loiter.d(numBlanco)*(elapsed_time_space.^3);
            aceleracion_real= blancos.acc(numBlanco) + loiter.c(numBlanco)*(elapsed_time_space) + loiter.d(numBlanco)*(elapsed_time_space.^2);
        
    end
    
    %% RANGE PLOTTING
    figure(1)
    subplot(2, blancos.N, numBlanco);
    ylabel("Distance (m)")
    xlabel("Elapsed time (s)")
    xlim([1, config.num_bursts])
    
    raw_data.r(raw_data.r == 0) = nan;
    raw_data.v(raw_data.v == 0) = nan;
    raw_data.a(raw_data.r == 0) = nan;
    
    tracked_data.r(tracked_data.r == 0) = nan;
    tracked_data.v(tracked_data.v == 0) = nan;
    tracked_data.a(tracked_data.r == 0) = nan;
    
    for numTraces = 1:min(size(tracked_data.r))
        zero_cells = tracked_data.r(:,numTraces)==0;
        
        error_aux = (trayectoria_real'-tracked_data.r(:,numTraces));
        error_aux(zero_cells) = nan;
        distance = (mean(error_aux.^2, 'omitnan'))^(0.5);
        
        % Si la trayectoria es parecida, se asigna la traza al blanco
        % estudiado
        if distance < likelihood_threshold_r
            traces_index = cat(2, traces_index, numTraces); %índices de las trazas que coinciden con el blanco
            % RMSE Trace
            RMSE_aux_r_trace = distance;
            RMSE_trace.r=cat(1, RMSE_trace.r, RMSE_aux_r_trace);
            tracked_error.r(:, numTraces)= error_aux;
            % Calculamos el error de la información sin tratar
            error_aux = (trayectoria_real'-raw_data.r(: ,numTraces));
            error_aux(zero_cells) = nan;
            RMSE_aux_r_raw = (mean(error_aux.^2, 'omitnan'))^(0.5);
            RMSE_raw.r=cat(1, RMSE_raw.r, RMSE_aux_r_raw);
            detected_error.r(:, numTraces)=error_aux;
                                 
            % Calculamos el error en velocidad
            % tracked
            error_aux = (velocidad_real'-tracked_data.v(: ,numTraces));
            error_aux(zero_cells) = nan;
            tracked_error.v(:, numTraces)=error_aux;
            RMSE_aux_v_trace = (mean(error_aux.^2, 'omitnan'))^(0.5);
            RMSE_trace.v=cat(1, RMSE_trace.v, RMSE_aux_v_trace);
            
            
            % Raw
            error_aux = (velocidad_real'-raw_data.v(: ,numTraces));
            error_aux(zero_cells) = nan;
            detected_error.v(:, numTraces)=error_aux;
            RMSE_aux_v_raw = (mean(error_aux.^2, 'omitnan'))^(0.5);
            RMSE_raw.v=cat(1, RMSE_raw.v, RMSE_aux_v_raw);  
            
            % Acceleration
            switch config.kalman_type
                case 0 % NCV
                    continue;
                case 1 % NCA
                    % tracked
                    error_aux = (aceleracion_real'-tracked_data.a(: ,numTraces));
                    error_aux(zero_cells) = nan;
                    tracked_error.a(:, numTraces)=error_aux;
                    RMSE_aux_a_trace = (mean(error_aux.^2, 'omitnan'))^(0.5);
                    RMSE_trace.a=cat(1, RMSE_trace.a, RMSE_aux_a_trace);


                    % Raw
                    error_aux = (aceleracion_real'-raw_data.a(: ,numTraces));
                    error_aux(zero_cells) = nan;
                    detected_error.a(:, numTraces)=error_aux;
                    RMSE_aux_a_raw = (mean(error_aux.^2, 'omitnan'))^(0.5);
                    RMSE_raw.a=cat(1, RMSE_raw.a, RMSE_aux_a_raw); 
            end
        end        
    end
    if (nnz(~isnan(tracked_data.r(:,traces_index(1)))==config.num_bursts))
        traces_index = traces_index(1);
    end
    hold on
    colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
    %% Tracked data
    tracked_data.r(tracked_data.r== 0) = nan;
    legend_aux=cell(1,length(traces_index)*2 + 1);
    for i = 1:length(traces_index)
        color = colors(i);
        plot(tracked_data.r(:,traces_index(i)), 'Marker','d','color', color, 'linewidth', 2)
        
        legend_aux{i} = strcat("Trace ",int2str(i)); 
    end
    xt = get(gca, 'XTick');                                
    set(gca, 'XTick', xt, 'XTickLabel', (xt-1)*config.burst_rep_period)
  
    
    
    %% Raw data
    raw_data.r(raw_data.r== 0) = nan;
    colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(raw_data.r(:,traces_index(i)),'color', color, 'linewidth', 1,'Marker','*')
        
        legend_aux{i+length(traces_index)} = strcat("Raw ",int2str(i));
    end
    
    
    plot(trayectoria_real, 'color', 'g', 'Marker','v', 'linewidth', 1.5)
    legend_aux{length(legend_aux)} = "Actual Distance";
    legend(legend_aux)
    hold off
    
    
    subplot(2, blancos.N, numBlanco+blancos.N);
    ylabel("Distance Square Error (m)")
    xlabel("Elapsed time (s)")
    %% Tracked error
    hold on
    legend_aux=cell(1,length(traces_index)*2);
    colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
    xlim([1, config.num_bursts])
    for i = 1:length(traces_index)
        color = colors(i);
        plot(tracked_error.r(:,traces_index(i)),'Marker','o','color', color, 'linewidth', 2)
        
        legend_aux{i} = strcat("RMSE = ",num2str(RMSE_trace.r(i)));
    end
    
    %% Raw data error
    
    colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(detected_error.r(:,traces_index(i)),'Marker','*','color', color, 'linewidth', 1.5)
        
        legend_aux{i+length(traces_index)} = strcat("RMSE = ",num2str(RMSE_raw.r(i)));
    end
    legend(legend_aux)
    hold off
    
    %% VELOCITY PLOTTING
    figure(2)
    tracked_data.v(tracked_data.v== 0) = nan;
    
    subplot(2, blancos.N, numBlanco);
    ylabel("Velocity (m/s)")
    xlabel("Elapsed time (s)")
    legend_aux=cell(1,length(traces_index)*2 + 1);
    xlim([1, config.num_bursts])
    hold on
    %% Tracked velocity
    colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(tracked_data.v(:,traces_index(i)), 'Marker','d', 'color', color, 'linewidth', 2)
        
        legend_aux{i} = strcat("Trace ",int2str(i));
    end 
    xt = get(gca, 'XTick');                                
    set(gca, 'XTick', xt, 'XTickLabel', (xt-1)*config.burst_rep_period)
       
    
    %% Raw velocity
    raw_data.v(raw_data.v== 0) = nan;
    colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(raw_data.v(:,traces_index(i)),'color', color, 'linewidth', 1,'Marker','*')
        
        legend_aux{i+length(traces_index)} = strcat("Raw ",int2str(i));
    end 
    plot(velocidad_real, 'Marker', 'v', 'color', 'g', 'linewidth', 1.5)
    legend_aux{length(legend_aux)} = "Actual Velocity";
    legend(legend_aux)
    hold off
    
    subplot(2, blancos.N, numBlanco+blancos.N);
    ylabel("Velocity Square Error (m)")
    xlabel("Elapsed time (s)")
    hold on
    %% Tracked error
    xlim([1, config.num_bursts])
    legend_aux=cell(1,length(traces_index)*2);
    colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(tracked_error.v(:,traces_index(i)),'Marker','o', 'color', color, 'linewidth', 2)
        
        legend_aux{i} = strcat("RMSE = ",num2str(RMSE_trace.v(i)));
    end
    %% Raw data error
    colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
    for i = 1:length(traces_index)
        color = colors(i);
        plot(detected_error.v(:,traces_index(i)),'Marker','*', 'color', color, 'linewidth', 1.5)
        
        legend_aux{i+length(traces_index)} = strcat("RMSE = ",num2str(RMSE_raw.v(i)));
    end
    legend(legend_aux)
    hold off
    
    %% Acceleration plot
    switch config.kalman_type
        case 0 % NCV
            continue;
        case 1 % NCA
            figure(3)
            tracked_data.a(tracked_data.r== 0) = nan;

            subplot(2, blancos.N, numBlanco);
            ylabel("Acceleration (m/s^2)")
            xlabel("Elapsed time (s)")
            legend_aux=cell(1,length(traces_index)*2 + 1);
            xlim([1, config.num_bursts])
            hold on
            %% Tracked acceleration
            colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
            for i = 1:length(traces_index)
                color = colors(i);
                plot(tracked_data.a(:,traces_index(i)), 'Marker','d', 'color', color, 'linewidth', 2)

                legend_aux{i} = strcat("Trace ",int2str(i));
            end 
            xt = get(gca, 'XTick');                                
            set(gca, 'XTick', xt, 'XTickLabel', (xt-1)*config.burst_rep_period)


            %% Raw acceleration
            raw_data.a(raw_data.r== 0) = nan;
            colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
            for i = 1:length(traces_index)
                color = colors(i);
                plot(raw_data.a(:,traces_index(i)),'color', color, 'linewidth', 1,'Marker','*')

                legend_aux{i+length(traces_index)} = strcat("Raw ",int2str(i));
            end 
            plot(aceleracion_real, 'Marker', 'v', 'color', 'g', 'linewidth', 1.5)
            legend_aux{length(legend_aux)} = "Actual Acceleration";
            legend(legend_aux)
            hold off

            subplot(2, blancos.N, numBlanco+blancos.N);
            ylabel("Acceleration Square Error (m/s^2)")
            xlabel("Elapsed time (s)")
            hold on
            %% Tracked error
            xlim([1, config.num_bursts])
            legend_aux=cell(1,length(traces_index)*2);
            colors = ["r" "#A2142F" "#7E2F8E" "#D95319" "#FF00FF" "m" "g" "y"];
            for i = 1:length(traces_index)
                color = colors(i);
                plot(tracked_error.a(:,traces_index(i)),'Marker','o', 'color', color, 'linewidth', 2)

                legend_aux{i} = strcat("RMSE = ",num2str(RMSE_trace.a(i)));
            end
            %% Raw data error
            
            colors = ["b" "#4DBEEE" "#0072BD" "#00FFFF" "#FFFF00" "m" "g" "y"];
            for i = 1:length(traces_index)
                color = colors(i);
                plot(detected_error.a(:,traces_index(i)),'Marker','*', 'color', color, 'linewidth', 1.5)

                legend_aux{i+length(traces_index)} = strcat("RMSE = ",num2str(RMSE_raw.a(i)));
            end
            legend(legend_aux)
            hold off
    end
    
end

disp('SIMULACIÓN FINALIZADA')
close(barra); 