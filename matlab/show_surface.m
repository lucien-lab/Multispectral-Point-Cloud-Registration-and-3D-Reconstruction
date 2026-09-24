

% 加载高斯模型数据
data = load('../gs_result_model.mat');

% 显示基本信息
fprintf('高斯点云信息:\n');
fprintf('  高斯数量: %d\n', data.gaussian_count);
fprintf('  位置维度: %d×%d\n', size(data.gaussian_positions));
fprintf('  颜色维度: %d×%d\n', size(data.gaussian_colors));

% 3D散点图显示高斯中心点
figure('Position', [100, 100, 1200, 800]);
scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), data.gaussian_positions(:,3), ...
         20, data.gaussian_colors, 'filled', 'MarkerEdgeColor', 'k');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Gaussian Splatting - 高斯中心点分布');
grid on;
axis equal;