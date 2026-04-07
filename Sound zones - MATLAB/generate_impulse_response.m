clear; clc;

save_path = 'room_dataset.h5';
if exist(save_path, 'file'), delete(save_path); end

%% parameters
N_rooms = 5000; % change to 20000 later
Fs = 48000; % sample rate
grid_step = 0.5; % receiver grid spacing (in meters)
src_margin = 0.5; % distance from walls for src and recvs (in meters)
T_fixed = round(0.5 * Fs); % RIR length is 0.5 seconds (adjust later)

% min/max of dimensions
x_min = 2; x_max = 10;
y_min = 2; y_max = 10;
z_min = 2; z_max = 5;
max_aspect = 3.0;

% "classes" of absorptions
alpha_classes = [0.05, 0.15, 0.3, 0.5, 0.7];

% fractional positions along each axis (2x2x2 = 8 sources)
fx = [1/4, 3/4];
fy = [1/4, 3/4];
fz = [1/3, 2/3];

rooms(N_rooms) = struct();

t_total = tic;

%% create rooms and estimate impulse responses
for i = 1:N_rooms

    t_room = tic;

    while true
        Lx = exp(log(x_min) + (log(x_max) - log(x_min)) * rand());
        Ly = exp(log(y_min) + (log(y_max) - log(y_min)) * rand());
        Lz = z_min + (z_max - z_min) * rand();

        if max(Lx, Ly) / min(Lx, Ly) <= max_aspect
            break;
        end
    end
    Lx = round(Lx, 2);
    Ly = round(Ly, 2);
    Lz = round(Lz, 2);

    % absorption
    alpha_idx = randi(length(alpha_classes));
    alpha = alpha_classes(alpha_idx);

    % place sources
    source_pos = zeros(8, 3);
    src_idx = 1;
    for xi = 1:2
        for yi = 1:2
            for zi = 1:2
                source_pos(src_idx, :) = [Lx*fx(xi), Ly*fy(yi), Lz*fz(zi)];
                src_idx = src_idx + 1;
            end
        end
    end

    % 3D grid of receivers
    rx_vec = src_margin : grid_step : Lx - src_margin;
    ry_vec = src_margin : grid_step : Ly - src_margin;
    rz_vec = 0.8 : grid_step : min(Lz - 0.3, 1.8);
    [RX, RY, RZ] = meshgrid(rx_vec, ry_vec, rz_vec);
    receiver_pos = [RX(:), RY(:), RZ(:)];
    N_rcv = size(receiver_pos, 1);

    fprintf('\n[Room %d/%d] dim=[%.2f x %.2f x %.2f]m  alpha=%.2f  receivers=%d\n', ...
        i, N_rooms, Lx, Ly, Lz, alpha, N_rcv);

    % preallocate — RIR length will vary slightly; grab length from first call
    rir_all = zeros(T_fixed, 8, N_rcv, 'single'); % will init after first call

    for s = 1:8
        t_src = tic;
        for r = 1:N_rcv
            rir = acousticRoomResponse( ...
                [Lx, Ly, Lz], ...
                source_pos(s, :), ...
                receiver_pos(r, :), ...
                Algorithm="image-source", ...
                ImageSourceOrder=2, ...
                SampleRate=Fs, ...
                MaterialAbsorption=alpha);

            rir = rir(:);
            T_rir = length(rir);

            if T_rir >= T_fixed
                rir_all(:, s, r) = single(rir(1:T_fixed));
            else
                rir_all(1:T_rir, s, r) = single(rir);
            end
        end

        fprintf('  source %d/8 done — %.1fs  (%d RIRs)\n', s, toc(t_src), N_rcv);
    end

    % store metadata
    rooms(i).room_dim = [Lx, Ly, Lz];
    rooms(i).alpha = alpha;
    rooms(i).source_pos = source_pos;
    rooms(i).receiver_pos = receiver_pos; 
    rooms(i).rir = rir_all;

    grp = sprintf('/room_%04d', i);

    h5create(save_path, [grp '/room_dim'], [1 3]);
    h5write(save_path, [grp '/room_dim'], rooms(i).room_dim);

    h5create(save_path, [grp '/alpha'], 1);
    h5write(save_path, [grp '/alpha'], rooms(i).alpha);

    h5create(save_path, [grp '/source_pos'], [8 3]);
    h5write(save_path, [grp '/source_pos'], rooms(i).source_pos);

    h5create(save_path, [grp '/receiver_pos'], [N_rcv 3]);
    h5write(save_path, [grp '/receiver_pos'], rooms(i).receiver_pos);

    h5create(save_path, [grp '/rir'], [T_fixed 8 N_rcv], ...
        "Datatype", "single", ...
        "ChunkSize", [T_fixed 1 1], ...
        "Deflate", 4);
    h5write(save_path, [grp '/rir'], rooms(i).rir);

    t_room_elapsed = toc(t_room);
    t_total_elapsed = toc(t_total);
    t_remaining = (t_total_elapsed / i) * (N_rooms - i);

    fprintf('  >> Room %d done in %.1fs | elapsed: %s | est. remaining: %s\n', ...
            i, t_room_elapsed, format_time(t_total_elapsed), format_time(t_remaining));
end

fprintf("All done :D Total time: %s\n", format_time(toc(t_total)));

function s = format_time(seconds)
    h = floor(seconds / 3600);
    m = floor(mod(seconds, 3600) / 60);
    s = floor(mod(seconds, 60));
    s = sprintf('%02d:%02d:%02d', h, m, s);
end