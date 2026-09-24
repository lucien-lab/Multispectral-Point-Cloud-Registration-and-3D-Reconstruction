%% 0. 环境清理 + 线程全开
clear all; clc; clf;
maxNumCompThreads(12);          % Ultra 5 吃满 12 线程
p = gcp('nocreate'); if isempty(p), parpool('local',12); end

%% 1. 原始常量（完全保留）
dy_datas   = load("../dy.txt");
dark       = load("../dark.txt");
white      = load("../white.txt");
plot(white(:,1), white(:,2));
white_max  = max(white(:, 2));
white_min  = min(white(:, 2));
angle_datas= load("../el_re.txt");

folder     = "../gp/specData";
files      = dir(folder);
fileNames  = {files(~[files.isdir]).name};
n          = numel(fileNames);

data2file  = fopen('../output_gp_dy.txt', 'w');

%% 2. 预分配 + 并行计算 RGB（原逻辑不动）
r_index = zeros(n,1,'single');
g_index = zeros(n,1,'single');
b_index = zeros(n,1,'single');
tmp1Vec = zeros(n,1,'uint32');
valid   = false(n,1);           % 记录哪些 i 最终要写入

parfor i = 1:n
    tmp1 = floor(i/268);
    if mod(tmp1,2) == 1
        continue;               % 原样跳过
    end
    type = 0;
    tmp_path = "../gp/specData/spec_" + num2str(i) + ".txt";
    gp_data  = load(tmp_path);

    index = 1;
    t     = max(gp_data(:,2));

    % 原 RGB 计算
    r = gp_data(957,2) / index;
    g = gp_data(730,2) / index;
    b = gp_data(552,2) / index;

    r_idx = (r - dark(957,2))/(white(957,2) - dark(957,2));
    g_idx = (g - dark(730,2))/(white(730,2) - dark(730,2));
    b_idx = (b - dark(552,2))/(white(552,2) - dark(552,2));

    % 保存结果
    r_index(i) = r_idx;
    g_index(i) = g_idx;
    b_index(i) = b_idx;
    tmp1Vec(i) = tmp1;
    valid(i)   = ismember(i, dy_datas(:,1));   % 是否要输出
end

%% 3. 顺序写文件（parfor 后统一写，避免并发冲突）
for i = 1:n
    if ~valid(i), continue; end
    fprintf(data2file, '%d,%d,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%6.3f,%d\n', ...
            tmp1Vec(i), i, dy_datas(ismember(dy_datas(:,1),i),2:4)', ...
            r_index(i), g_index(i), b_index(i), 0);
end
fclose(data2file);

%% 4. 画图（与原脚本一致）
data_color = load("../output_gp_dy.txt");
scatter3(-1.4*data_color(:,3), 1.4*data_color(:,4), data_color(:,5)*0.7, ...
         10, data_color(:,6:8)*4, 'filled');
axis equal;
