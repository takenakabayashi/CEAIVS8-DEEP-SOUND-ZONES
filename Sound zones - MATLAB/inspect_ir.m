clear; clc;

room_name = "ListeningRoom";
src_idx = 1;
height = 1.0;
idxX_pick = 16;
idxY_pick = 16;

filename = sprintf('ISOBEL_RTFs/%s_RTFs/source_%d/h_%d/idxX_%d_idxY_%d.mat', ...
    room_name, src_idx, height * 100, idxX_pick, idxY_pick);

d = load(filename);
RTF = double(d.RTF(:));
freqs = double(d.freqs(:));
rx_pos = d.rx_pos;

% flipping time 
RTF = conj(RTF);

K = numel(RTF);
Fs = 1000; % sampling freq
duration = 1.0;
N = 2 * K;
df = Fs / N;

% 3. Build Hermitian spectrum (your construction is correct)
RTF(1) = real(RTF(1)); % enforcing real DC
H_full = [RTF; 0; conj(flipud(RTF(2:end)))];
h = ifft(H_full, 'symmetric');

fprintf('Receiver pos: [%.2f, %.2f, %.2f]\n', rx_pos);
fprintf('IR length: %d samples (%.3f s at Fs=%d Hz)\n', length(h), length(h) / Fs, Fs);
fprintf('Max |imag|: %.2e\n', max(abs(imag(ifft(H_full)))));
fprintf('Max |h|: %.4e\n', max(abs(h)));
fprintf('Mean(h): %.4e\n', mean(h));
fprintf('Any NaN/Inf: %d\n', any(~isfinite(h)));

[~, kpk] = max(abs(h));
fprintf('Peak at: sample %d (%.1f ms)\n', kpk, (kpk - 1) / Fs * 1000);

% plots
t = (0:N-1).' / Fs;

figure('Position', [100 100 1000 700]);

% time-domain IR
subplot(3,1,1);
plot(t, h, 'b'); hold on;
% expected exponential decay envelope from T60
T60 = 0.6;
delta = 3 * log(10) / T60;
[pk, kpk] = max(abs(h));
t_pk = (kpk - 1) / Fs;
env = pk * exp(-delta * (t - t_pk));
env(t < t_pk) = NaN;
plot(t, env, 'r--', 'LineWidth', 1);
plot(t, -env, 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Amplitude');
title(sprintf('Impulse response — %s, src %d, rcv [%.1f, %.1f, %.1f]', ...
    room_name, src_idx, rx_pos));
legend('h(t)', sprintf('\\pm A_{peak}·exp(-(t-t_{peak})·3ln10/T60), T60=%.2fs', T60), ...
       'Location', 'northeast');
grid on;

% zoomed in onset
subplot(3,1,2);
src_pos = [0.32, 0.22, 0.15]; % adjust to actual source you picked
direct_delay = norm(rx_pos - src_pos) / 343;
plot(t, h, 'b'); hold on;
xline(direct_delay, 'g--', 'direct path');
xlim([0, 0.1]);
xlabel('Time [s]'); ylabel('Amplitude');
title(sprintf('First 100 ms — direct path expected at %.1f ms', direct_delay*1000));
grid on;

% magnitude spectrum check (should match RTF magnitude)
subplot(3,1,3);
H_check = fft(h);
plot(freqs, 20*log10(abs(RTF) + eps), 'b', 'LineWidth', 1.2); hold on;
plot(freqs, 20*log10(abs(H_check(1:K)) + eps), 'r--');
xlabel('Frequency [Hz]'); ylabel('|H(f)| [dB]');
title('Spectrum sanity check (blue: original |RTF|, red dashed: |FFT(h)|)');
legend('original |RTF|', '|FFT(h)|', 'Location', 'southwest');
grid on; xlim([0 Fs/2]);