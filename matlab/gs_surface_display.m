% gs_surface_display.m - 显示Gaussian Splatting生成的曲面
function gs_surface_display(filename, resolution_factor)
    if nargin < 2
        resolution_factor = 1;  % 控制网格分辨率
    end
    
    % 加载高斯模型数据
    data = load(filename);
    
    fprintf('=== Gaussian Splatting Surface Display ===\n');
    fprintf('高斯数量: %d\n', data.gaussian_count);
    
    % 创建规则网格用于曲面重建
    [x_range, y_range, z_range] = get_bounding_box(data.gaussian_positions);
    
    % 根据分辨率因子调整网格密度
    grid_density = 50 * resolution_factor;
    x_grid = linspace(x_range(1), x_range(2), grid_density);
    y_grid = linspace(y_range(1), y_range(2), grid_density);
    z_grid = linspace(z_range(1), z_range(2), grid_density);
    
    % 创建2D网格（XY平面投影）
    [X, Y] = meshgrid(x_grid, y_grid);
    Z = zeros(size(X));
    
    fprintf('正在计算曲面...\n');
    
    % 高斯混合模型计算（简化版曲面重建）
    for i = 1:size(X, 1)
        for j = 1:size(X, 2)
            point = [X(i,j), Y(i,j), median(z_grid)];  % 使用中位数Z值作为初始猜测
            density = 0;
            
            % 计算该点的高斯密度贡献
            for k = 1:data.gaussian_count
                pos = data.gaussian_positions(k, :);
                cov_inv = inv(data.covariance_matrices{k});  % 假设有协方差矩阵
                diff = point - pos;
                exponent = -0.5 * diff * cov_inv * diff';
                density = density + data.gaussian_opacities(k) * exp(exponent);
            end
            
            Z(i,j) = density;
        end
        fprintf('进度: %.1f%%\r', i/size(X,1)*100);
    end
    
    % 创建3D曲面显示
    figure('Position', [100, 100, 1600, 1200]);
    
    % 3D曲面图
    subplot(2,2,1);
    surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
    title('Gaussian Splatting Surface (XY Projection)');
    xlabel('X'); ylabel('Y'); zlabel('Density');
    colorbar;
    axis equal;
    
    % 等高线图
    subplot(2,2,2);
    contour(X, Y, Z, 20);
    title('Surface Contours');
    xlabel('X'); ylabel('Y');
    axis equal;
    colorbar;
    
    % 3D点云叠加显示
    subplot(2,2,3);
    scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), data.gaussian_positions(:,3), ...
             20, data.gaussian_colors, 'filled', 'Alpha', 0.6);
    hold on;
    % 在相同位置显示重建的曲面点
    [surf_x, surf_y] = meshgrid(linspace(x_range(1), x_range(2), 30), ...
                               linspace(y_range(1), y_range(2), 30));
    surf_z = interp2(X, Y, Z, surf_x, surf_y);
    scatter3(surf_x(:), surf_y(:), surf_z(:), 10, 'r', 'filled', 'Alpha', 0.8);
    title('Points + Reconstructed Surface');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    legend('Original Points', 'Reconstructed Surface');
    
    % 密度分布直方图
    subplot(2,2,4);
    histogram(Z(:), 50);
    title('Surface Density Distribution');
    xlabel('Density Value'); ylabel('Frequency');
    grid on;
    
    sgtitle(sprintf('Gaussian Splatting Surface Reconstruction (%d Gaussians)', data.gaussian_count));
    
    % 保存结果
    print('gs_surface_result.png', '-dpng', '-r300');
    fprintf('结果已保存为 gs_surface_result.png\n');
end

function [x_range, y_range, z_range] = get_bounding_box(positions)
    padding = 0.1;  % 添加边距
    x_range = [min(positions(:,1)) - padding, max(positions(:,1)) + padding];
    y_range = [min(positions(:,2)) - padding, max(positions(:,2)) + padding];
    z_range = [min(positions(:,3)) - padding, max(positions(:,3)) + padding];
end