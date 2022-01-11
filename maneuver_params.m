%%
close all
for i = [-1/10 -1/2 -1 -2 -10 1/10 1/2 0 1 2 10]
    for j =[-1/10 -1/2 -1 -2 -10 1/10 1/2 0 1 2 10]
        x = -1000;
        v = 300;
        b = 100;

        c = b/i;
        d = b/j;
%         c=b*10;
%         d=b*-50;

        t = linspace(0,2, 100);

        v_t = (v+b*t)+c*(0.5*t.^2) + ((1/10)*d*t.^2);

        figure
        plot(t, v_t)
        leg= strcat("i", num2str(i),"j", num2str(j));
        legend(leg)
    end
end