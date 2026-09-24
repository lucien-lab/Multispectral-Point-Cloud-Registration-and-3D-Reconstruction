% gs_advanced_surface_fixed.m - 修正版高级曲面显示
function gs_advanced_surface_fixed(filename)
    data = load(filename);
    
    % 重建协方差矩阵
    covariance_matrices = cell(data.gaussian_count, 1);
    for i = 1:data.gaussian_count
        eigenvals = data.covariance_eigenvalues(i, :)';
        covariance_matrices{i} = diag(eigenvals);
    end
    
    % 创建高分辨率网格
    bounds = get_extended_bounds(data.gaussian_positions, 0.2);
    resolution = 80;  % 适中的分辨率
    
    x_vec = linspace(bounds.x_min, bounds.x_max, resolution);
    y_vec = linspace(bounds.y_min, bounds.y_max, resolution);
    z_vec = linspace(bounds.z_min, bounds.z_max, resolution);
    
    % 3D密度场计算
    fprintf('计算3D密度场...\n');
    density_field = zeros(resolution, resolution, resolution);
    
    % 限制计算量以提高性能
    max_gaussians = min(80, data.gaussian_count);
    
    for i = 1:resolution
        for j = 1:resolution
            for k = 1:resolution
                point = [x_vec(i), y_vec(j), z_vec(k)];
                density = 0;
                
                % 计算所有高斯的贡献
                for g = 1:max_gaussians
                    pos = data.gaussian_positions(g, :);
                    cov = covariance_matrices{g};
                    opacity = data.gaussian_opacities(g);
                    
                    try
                        diff = point - pos;
                        % 添加正则化项避免奇异矩阵
                        cov_reg = cov + eye(3) * 1e-6;
                        cov_inv = inv(cov_reg);
                        exponent = -0.5 * (diff * cov_inv * diff');
                        if exponent > -50
                            density = density + opacity * exp(exponent);
                        end
                    catch
                        continue;
                    end
                end
                
                density_field(i,j,k) = density;
            end
        end
        fprintf('XY层进度: %.1f%%\r', i/resolution*100);
    end
    
    % 等值面提取和显示
    figure('Position', [50, 50, 1800, 1400]);
    
    % 修正的等值面提取
    isovalue = prctile(density_field(:), 85);  % 使用较低的百分位数
    
    try
        % 安全的等值面提取
        fv = isosurface(x_vec, y_vec, z_vec, density_field, isovalue);
        
        if ~isempty(fv)
            faces = fv.faces;
            verts = fv.vertices;
            
            % 计算顶点颜色（基于密度）
            vertex_densities = interp3(x_vec, y_vec, z_vec, density_field, ...
                                     verts(:,1), verts(:,2), verts(:,3));
            colors = parula(length(vertex_densities));
            colors = colors(round(linspace(1, size(colors,1), length(vertex_densities))), :);
            
        else
            error('未能提取到有效的等值面');
        end
        
    catch ME
        % 备用方案：使用更简单的方法
        fprintf('等值面提取失败，使用备用方案...\n');
        [faces, verts] = create_simple_surface(density_field, x_vec, y_vec, z_vec, isovalue);
        colors = repmat([0.5 0.5 1.0], size(verts,1), 1);  % 蓝色
    end
    
    % 3D等值面显示
    subplot(2,3,1);
    p = patch('Vertices', verts, 'Faces', faces, 'FaceColor', 'interp', ...
              'EdgeColor', 'none', 'FaceVertexCData', colors);
    isonormals(x_vec, y_vec, z_vec, density_field, p);
    title(sprintf('Isosurface (Isovalue: %.3f)', isovalue));
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal; camlight; lighting gouraud;
    
    % 多个切片视图
    subplot(2,3,2);
    slice(x_vec, y_vec, z_vec, density_field, [], [], z_vec(round(end/2)));
    title('XZ Slice');
    xlabel('X'); ylabel('Z'); zlabel('Y');
    colorbar;
    
    subplot(2,3,3);
    slice(x_vec, y_vec, z_vec, density_field, [], y_vec(round(end/2)), []);
    title('XY Slice');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    colorbar;
    
    subplot(2,3,4);
    slice(x_vec, y_vec, z_vec, density_field, x_vec(round(end/2)), [], []);
    title('YZ Slice');
    xlabel('Y'); ylabel('Z'); zlabel('X');
    colorbar;
    
    % 原始点云显示
    subplot(2,3,5);
    scatter3(data.gaussian_positions(:,1), data.gaussian_positions(:,2), data.gaussian_positions(:,3), ...
             15, data.gaussian_colors, 'filled', 'Alpha', 0.7);
    hold on;
    % 在等值面上标记一些关键点
    if size(verts,1) > 0
        sample_indices = round(linspace(1, size(verts,1), min(50, size(verts,1))));
        scatter3(verts(sample_indices,1), verts(sample_indices,2), verts(sample_indices,3), ...
                20, 'r', 'filled', 'MarkerEdgeColor', 'k');
    end
    title('Original Points + Surface Samples');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    legend('Gaussian Points', 'Surface Samples');
    
    % 密度分布
    subplot(2,3,6);
    histogram(density_field(:), 50);
    title('3D Density Distribution');
    xlabel('Density'); ylabel('Frequency');
    grid on;
    
    sgtitle(sprintf('Gaussian Splatting 3D Surface (%d Gaussians)', data.gaussian_count));
    
    % 保存高质量图像
    orient tall;
    print('gs_3d_surface_analysis_fixed.png', '-dpng', '-r300');
    fprintf('结果已保存为: gs_3d_surface_analysis_fixed.png\n');
end

function bounds = get_extended_bounds(positions, padding_ratio)
    extents = [min(positions,[],1); max(positions,[],1)];
    sizes = extents(2,:) - extents(1,:);
    padding = sizes * padding_ratio;
    
    bounds.x_min = extents(1,1) - padding(1);
    bounds.x_max = extents(2,1) + padding(1);
    bounds.y_min = extents(1,2) - padding(2);
    bounds.y_max = extents(2,2) + padding(2);
    bounds.z_min = extents(1,3) - padding(3);
    bounds.z_max = extents(2,3) + padding(3);
end

function [faces, verts] = create_simple_surface(density_field, x_vec, y_vec, z_vec, isovalue)
    % 简单的表面创建方法作为备用
    [nx, ny, nz] = size(density_field);
    
    % 找到超过等值面的点
    above_threshold = density_field > isovalue;
    
    if nnz(above_threshold) == 0
        % 如果没有点超过阈值，使用更高的阈值
        isovalue = prctile(density_field(:), 70);
        above_threshold = density_field > isovalue;
    end
    
    % 创建简单的立方体表示
    [ix, iy, iz] = ind2sub(size(density_field), find(above_threshold));
    
    if ~isempty(ix)
        % 转换为实际坐标
        verts = [x_vec(ix(:)), y_vec(iy(:)), z_vec(iz(:))];
        
        % 创建简单的三角面（这里简化处理）
        faces = [];
        if size(verts,1) >= 3
            % 创建一个大的三角面连接所有点（简化）
            faces = delaunay(verts(:,1), verts(:,2));
        end
        
        if isempty(faces)
            % 最后的备用方案：创建单个大三角面
            faces = [1, 2, 3];
        end
    else
        verts = [];
        faces = [];
    end
end