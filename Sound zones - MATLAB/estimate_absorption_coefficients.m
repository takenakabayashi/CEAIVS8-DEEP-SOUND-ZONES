clear; clc;

Lx = 6.98;
Ly = 8.12;
Lz = 3.03;

V = Lx * Ly * Lz;
fprintf('Room volume %.2f m^3\n', V);

S_floor = Lx * Ly;
S_ceiling = Lx * Ly;
S_walls = 2 * Lx * Lz + 2 * Ly * Lz;
S_total = S_floor + S_ceiling + S_walls; % total surface area

% T20 measurements (from VRLab measurement report)
% values are read from figure 7 in that measurement report

fc = [50, 63, 79, 100, 126, 158, 200, 251, 316, 398, 501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000];
T20 = [0.59, 0.62, 0.41, 0.4, 0.3, 0.31, 0.3, 0.25, 0.25, 0.22, 0.21, 0.2, 0.21, 0.22, 0.21, 0.23, 0.22, 0.21, 0.21, 0.2, 0.25, 0.2, 0.35, 0.45];

n_bands = length(fc);

RT60 = 3 * T20;
alpha_sabine = (0.161 * V) ./ (S_total .* RT60);
fprintf("Sabine Alpha: %.2f\n", alpha_sabine);

alpha_eyring = 1 - exp(-(0.161 *V) ./ (S_total .* RT60));

RT60_predicted = -0.161 * V ./ (S_total .* log(1 - alpha_eyring));
T20_predicted = RT60_predicted / 3;

fprintf("Absorption coefficients across all surfaces:\n");
fprintf("%-8s %-10s %-12s %-14s %-10s %-10s %-10s\n", 'fc [Hz]', 'T20 [s]', 'RT60 [s]', 'RT60 pred [s]', 'T20 pred [s]', 'alpha Sabine', 'alpha Eyring');
for i = 1:n_bands
    fprintf('%-8d %-10.4f %-12.4f %-14.4f %-12.4f %-10.4f %-10.4f\n', fc(i), T20(i), RT60(i), RT60_predicted(i), T20_predicted(i), alpha_sabine(i), alpha_eyring(i));
end

figure('Position', [100 100 1200 450]);

subplot(1,3,1);
semilogx(fc, T20, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'T20 Mean');
xlabel('Frequency [Hz]'); 
ylabel('T20 [s]');
title('T20');
legend('Location', 'northeast');
grid on;
xlim([80 6000]);

subplot(1,3,2);
semilogx(fc, T20, 'k-o', 'LineWidth', 1.5, 'DisplayName', 'T20 input');
hold on;
semilogx(fc, T20_predicted, 'r--s', 'LineWidth', 1.5, 'DisplayName', 'T20 predicted');
xlabel('Frequency [Hz]');
ylabel('T20 [s]');
title('T20 Verification');
legend;
grid on;
xlim([80 6000]);

subplot(1,3,3);
semilogx(fc, alpha_sabine, 'k-o', 'LineWidth', 1.5, 'DisplayName', 'Alpha Sabine');
hold on;
semilogx(fc, alpha_eyring, 'r--s', 'LineWidth', 1.5, 'DisplayName', 'Alpha Eyring');
xlabel('Frequency [Hz]');
ylabel('Absorption coefficient \alpha');
title("Estimated uniform absorption coefficient");
grid on;
xlim([80 6000]);
ylim([0 1]);

absorption.fc = fc;
absorption.alpha_sabine = alpha_sabine;
absorption.alpha_eyring = alpha_eyring;
absorption.T20 = T20;
absorption.RT60 = RT60;
absorption.room_dim = [Lx, Ly, Lz];
absorption.S_total = S_total;
absorption.V = V;

save('absorption_coefficients.mat', 'absorption');