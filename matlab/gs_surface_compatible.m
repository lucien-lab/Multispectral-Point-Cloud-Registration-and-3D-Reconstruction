% gs_surface_compatible.m - 兼容所有MATLAB版本的曲面显示
function gs_surface_compatible(filename)
    % 检查输入文件
    if ~exist(filename, 'file')
        error(['文件不存在: ' filename]);
    end
    
    data = load(filename);
    fprintf('=== Gaussian Splatting Surface Analysis ===\n');
    fprintf('高斯数量: %d\n', data.gaussian_count);
    
    % 创建图形窗口
    figure('Position', [50, 50, 1600, 1200], 'Name', 'Gaussian Splatting Analysis');
    
    % 子图1: 原始点云（兼容版本）
    subplot(2,3,1);
    h1 = scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), ...
                  data.gaussian_positions(:,3), 20, data.gaussian_colors, 'filled');
    % 兼容性处理：检查是否支持Alpha属性
    if verLessThan('matlab', '8.4')  % R2014b之前版本
        % 老版本不支持Alpha，使用marker face transparency
        set(h1, 'MarkerFaceAlpha', 0.7);
    else
        % 新版本支持Alpha属性
        try
            set(h1, 'Alpha', 0.7);
        catch
            set(h1, 'MarkerFaceAlpha', 0.7);
        end
    end
    title(sprintf('Original Points (%d)', data.gaussian_count));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; grid on;
    
    % 子图2: 密度热力图
    subplot(2,3,2);
    % 计算2D密度分布
    nbins = 80;
    [N, edges] = histcounts2(data.gaussian_positions(:,1), data.gaussian_positions(:,2), nbins);
    x_centers = (edges{1}(1:end-1) + edges{1}(2:end)) / 2;
    y_centers = (edges{2}(1:end-1) + edges{2}(2:end)) / 2;
    [X, Y] = meshgrid(x_centers, y_centers);
    
    % 兼容性处理：使用contourf而非contour3
    contourf(X, Y, N', 20);
    colorbar;
    title('2D Density Map');
    xlabel('X'); ylabel('Y');
    axis equal;
    
    % 子图3: 按透明度着色的点云
    subplot(2,3,3);
    h3 = scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), ...
                  data.gaussian_positions(:,3), 20, data.gaussian_opacities, 'filled');
    if verLessThan('matlab', '8.4')
        set(h3, 'MarkerFaceAlpha', 0.7);
    else
        try
            set(h3, 'Alpha', 0.7);
        catch
            set(h3, 'MarkerFaceAlpha', 0.7);
        end
    end
    colorbar;
    title('Points by Opacity');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; grid on;
    
    % 子图4: 局部密度分析
    subplot(2,3,4);
    % 计算局部密度（简化版本）
    local_density = calculate_local_density(data.gaussian_positions, 15);
    h4 = scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), ...
                  data.gaussian_positions(:,3), 25, local_density, 'filled');
    if verLessThan('matlab', '8.4')
        set(h4, 'MarkerFaceAlpha', 0.8);
    else
        try
            set(h4, 'Alpha', 0.8);
        catch
            set(h4, 'MarkerFaceAlpha', 0.8);
        end
    end
    colorbar;
    title('Local Density Analysis');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    
    % 子图5: 统计分布图
    subplot(2,3,5);
    % 创建多合一统计图
    subplot(2,3,5);
    % 透明度分布
    histogram(data.gaussian_opacities, 25);
    title('Opacity Distribution');
    xlabel('Opacity Value'); ylabel('Frequency');
    grid on;
    
    % 子图6: 综合信息显示
    subplot(2,3,6);
    axis off;
    % 创建统计信息文本
    stats_info = create_statistics_text(data, local_density);
    text(0.1, 0.9, stats_info, 'FontSize', 11, 'VerticalAlignment', 'top', ...
         'FontName', 'Consolas', 'BackgroundColor', [0.95 0.95 0.95]);
    
    % 整体标题
    sgtitle(sprintf('Gaussian Splatting Surface Analysis (%d Points)', data.gaussian_count));
    
    % 保存结果（符合规范）
    output_filename = [filename(1:end-4), '_analysis.png'];
    orient tall;
    print(output_filename, '-dpng', '-r300');
    fprintf('分析结果已保存为: %s\n', output_filename);
end

function local_density = calculate_local_density(positions, k)
    % 计算局部密度的兼容版本
    n_points = size(positions, 1);
    local_density = zeros(n_points, 1);
    
    % 简化版本：使用距离的倒数作为密度指标
    for i = 1:min(100, n_points)  % 限制计算量
        % 计算到其他点的距离
        distances = sqrt(sum((positions - repmat(positions(i,:), n_points, 1)).^2, 2));
        % 排序并取前k个最近邻
        [sorted_distances, ~] = sort(distances);
        % 使用平均距离的倒数作为密度指标
        local_density(i) = 1 / (mean(sorted_distances(2:k+1)) + eps);
    end
    
    % 对于未计算的点，使用平均值
    if n_points > 100
        mean_density = mean(local_density(1:100));
        local_density(101:end) = mean_density;
    end
end

function stats_text = create_statistics_text(data, local_density)
    % 创建统计信息文本
    x_range = [min(data.gaussian_positions(:,1)), max(data.gaussian_positions(:,1))];
    y_range = [min(data.gaussian_positions(:,2)), max(data.gaussian_positions(:,2))];
    z_range = [min(data.gaussian_positions(:,3)), max(data.gaussian_positions(:,3))];
    
    stats_text = sprintf([...
        'Gaussian Splatting Statistics:\n\n' ...
        'Points Count: %d\n' ...
        'Coordinate Range:\n' ...
        '  X: [%.2f, %.2f] (span: %.2f)\n' ...
        '  Y: [%.2f, %.2f] (span: %.2f)\n' ...
        '  Z: [%.2f, %.2f] (span: %.2f)\n\n' ...
        'Opacity Stats:\n' ...
        '  Mean: %.3f\n' ...
        '  Range: [%.3f, %.3f]\n\n' ...
        'Density Stats:\n' ...
        '  Max Local: %.2f\n' ...
        '  Mean Local: %.2f'], ...
        data.gaussian_count, ...
        x_range(1), x_range(2), x_range(2)-x_range(1), ...
        y_range(1), y_range(2), y_range(2)-y_range(1), ...
        z_range(1), z_range(2), z_range(2)-z_range(1), ...
        mean(data.gaussian_opacities(:)), ...
        min(data.gaussian_opacities(:)), max(data.gaussian_opacities(:)), ...
        max(local_density), mean(local_density));
end

% 简化的快速调用函数
function quick_gs_display(filename)
    % 快速显示函数
    if ~exist(filename, 'file')
        fprintf('错误: 文件 %s 不存在\n', filename);
        return;
    end
    
    try
        gs_surface_compatible(filename);
        fprintf('✓ 显示完成!\n');
    catch ME
        fprintf('✗ 显示出错: %s\n', ME.message);
        % 提供简化版本作为备选
        fprintf('尝试简化版本...\n');
        gs_simple_display(filename);
    end
end

function gs_simple_display(filename)
    % 极简版本显示
    data = load(filename);
    
    figure('Position', [100, 100, 1200, 800]);
    
    scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), ...
             data.gaussian_positions(:,3), 15, data.gaussian_colors, 'filled');
    
    title(sprintf('Gaussian Splatting Points (%d)', data.gaussian_count));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; grid on;
    
    print([filename(1:end-4), '_simple.png'], '-dpng', '-r300');
end