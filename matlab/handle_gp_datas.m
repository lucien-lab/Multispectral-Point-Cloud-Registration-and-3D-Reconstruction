%% 0. 环境清理 + 线程全开
clear all; clc;
maxNumCompThreads(12);
p = gcp('nocreate'); if isempty(p), parpool('local',12); end

%% 1. 原始常量（不动）
folder      = "../sbq/";
files       = dir(folder);
fileNames   = {files(~[files.isdir]).name};
n           = numel(fileNames);
angle_datas = load("../el_re.txt");
data2file   = fopen('../dy.txt', 'w');

%% 2. 预分配 + 并行计算坐标（原逻辑一字不改）
XYZ     = zeros(n, 3,'single');      % 存结果
dVec    = zeros(n,1,'single');       % 存距离，供后面写文件
paramV  = zeros(n,1,'uint16');       % 存 param
flagV   = false(n,1);                % 哪些 i 真正被处理

parfor i = 1:n
    if contains(fileNames(1, i), ".txt")
        continue
    end
    tmp_path   = num2str(i) + ".csv";
    combinedStr= append(folder, tmp_path);
    if ~isfile(combinedStr)
        continue
    end
    RawData = load(combinedStr);
    if size(RawData,1) ~= 1000
        continue
    end
    [~, ~, d] = Processing(RawData(2:end), tmp_path, i);
    if d == 10
        continue
    end
    
    param = 0;
    horizontal_angle = deg2rad(angle_datas(i-param, 2));
    altitude_angle   = deg2rad(angle_datas(i-param, 3));
    if mod(i,2)==0
        horizontal_angle = deg2rad(angle_datas(i-param, 2));
    end
    
    X_cor = d * cos(altitude_angle) * sin(horizontal_angle);
    Y_cor = d * cos(altitude_angle) * cos(horizontal_angle);
    Z_cor = d * sin(altitude_angle);
    [x_new, y_new, z_new] = rotate_around_y(X_cor, Y_cor, Z_cor, 180);
    
    % 保存结果
    XYZ(i,:)   = [x_new, y_new, z_new];
    dVec(i)    = d;
    paramV(i)  = param;
    flagV(i)   = true;
end

%% 3. 顺序写文件（避免并发冲突）
for i = 1:n
    if ~flagV(i), continue; end
    fprintf(data2file, '%d,%6.6f,%6.6f,%6.6f\n', i, XYZ(i,1), XYZ(i,2), XYZ(i,3));
end
fclose(data2file);

%% 4. 画图（与原脚本一致）
plot3(XYZ(:,1), XYZ(:,2), XYZ(:,3), '.b');
axis equal;
ylim([1 1.8]); xlim([-0.2 0.4]); zlim([-0.4 0]);
