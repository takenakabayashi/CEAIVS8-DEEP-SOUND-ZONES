clear; clc;

x_min = 2; % min possible x
x_max = 10; % max possible x

y_min = 2; % min possible y
y_max = 10; % max possible y

z_min = 2; % min z
z_max = 5; % max z


for i = 1:10
    Lx = round(x_min + (x_max - x_min) * rand(), 2);
    Ly = round(y_min + (y_max - y_min) * rand(), 2);
    Lz = round(z_min + (z_max - z_min) * rand(), 2);
    fprintf('ROOM DIM: [%d, %d, %d]\n', Lx, Ly, Lz);
end