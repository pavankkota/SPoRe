function l2err = group_fro(Y, X, Phis, group_indices)
    l2err = 0; 
    for g = 1:max(group_indices)
        l2err = l2err + norm(Y(:,group_indices==g) - Phis(:,:,g)*X(:,group_indices==g), 'fro');
    end
end
