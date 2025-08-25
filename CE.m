function [U, obj] = CE(X, v, n, dd, c, alpha, beta, eta, alpha1, options, gamma)

%% ===================== initialize =====================
% MaxIter = 15;
MaxIter = 50; % only BBCsport 50 iter

S = cell(1,v);
Lopt = zeros(n,n);
U = cell(1,v);
Xv_bar = X;
Z = cell(1,v);
Dv = cell(1,v);  % update U
for i = 1:v
    Z{i} = ones(n,n);
    U{i} = ones(dd(i),c);
    Dv{i} = eye(dd(i));
end

% initalize tensor
J = cell(1,v);
J_tensor = cell(1,v);
for i = 1:v
    J{i} = zeros(n,n);
    J_tensor = zeros(n,n);
end

%% ===================== updating =====================
for iter = 1:MaxIter
    % update U
    for iterv = 1:v
        LG = (eye(n) - Z{iterv});
        LG = LG * LG';
        LG = (LG + LG') / 2;
        [Y, ~, ~]=eig1(LG, c, 0);
        U{iterv}=(Xv_bar{iterv}*Xv_bar{iterv}'+alpha*Dv{iterv})\(Xv_bar{iterv}*Y);
        Ui=sqrt(sum(U{iterv}.*U{iterv},2)+eps);
        diagonal=0.5./Ui;
        Dv{iterv}=diag(diagonal);
    end

    % update Xv_bar ------ use Z{v}
    for iterv = 1:v
        Xv_bar_temp = X{iterv}';
        Lz = diag(sum(Z{iterv}) - Z{iterv});
        Xv_bar_temp = (eta*(eye(n)-Lz/2)+(1-eta)*(Lz/2))*Xv_bar_temp;
        Xv_bar{iterv} = Xv_bar_temp';
    end

    % update Z
    for iterv = 1:v
        temp1 = Xv_bar{iterv}'*U{iterv}*U{iterv}'*Xv_bar{iterv} + beta*eye(n) + gamma*Lopt;
        temp2 = Xv_bar{iterv}'*U{iterv}*U{iterv}'*Xv_bar{iterv} + beta*J{iterv};
        temp = temp1\temp2;
        Z0 = zeros(size(temp));
        for is = 1:size(temp,1)
            ind_c = 1:size(temp,1);
            ind_c(is) = [];
            Z0(is,ind_c) = EProjSimplex_new(temp(is,ind_c));
        end
        Z{iterv} = Z0;
    end

    % update L
    for iv = 1:v
        S{iv} = constructW(Xv_bar{iv}', options);
        L{iv} = constructL(S{iv}, alpha1, options);
    end
    for iv = 1:v
        S{iv} = full(S{iv});
    end
    for iv = 1:v
        Sopt(:,:,iv) = S{iv};
    end
    Sopt = min(Sopt,[],3);
    Lopt = constructL(Sopt, alpha1, options);
    
    % update tensor
    Z_tensor = cat(3, Z{:,:});
    hatZ = fft(Z_tensor, [], 3);
    if iter == 1
        for iv = 1:v
            [Unum_view, Sigmanum_view, Vnum_view] = svds(hatZ(:,:,iv), c);
            T{iv} = Unum_view * Sigmanum_view;
            V{iv} = Vnum_view';
            J_tensor(:,:,iv) = T{iv} * V{iv};
        end
    else
        for iv = 1:v
            T{iv} = hatZ(:,:,iv) * V{iv}' * pinv(V{iv} * V{iv}');
            Usq{iv} = T{iv}' * T{iv};
            V{iv} = pinv(Usq{iv}) * T{iv}' * hatZ(:,:,iv);
            J_tensor(:,:,iv) = T{iv} * V{iv};
        end
    end
    J_tensor = ifft(J_tensor, [], 3);
    for iv = 1 : v
        J{iv} = J_tensor(:,:,iv);
    end

    %% ===================== calculate obj =====================
    Term1 = 0;
    Term2 = 0;
    Term3 = 0;
    for objIndex = 1:v
        Term1 = Term1 + norm(U{objIndex}'*Xv_bar{objIndex}-U{objIndex}'*Xv_bar{objIndex}*Z{objIndex}, 'fro').^2;
        Term2 = Term2 + alpha*sum(sqrt(sum(U{objIndex}.*U{objIndex},2)));
        Term3 = Term3 + gamma*trace(Z{objIndex}'*Lopt*Z{objIndex});
    end
    tempobj = Term1 + Term2 + Term3;

    % calculate tensor
    for iv = 1:v
        leq{iv} = Z{iv}-J{iv};
    end
    leqm = cat(3,leq{:,:});
    leqm2 = max(abs(leqm(:)));
    tensor_err = max(leqm2);
    Term5 = beta*tensor_err;

    tempobj = tempobj + Term5;

    obj(iter) = tempobj;
    if iter == 1
        err = 0;
    else
        err = obj(iter)-obj(iter-1);
    end

    fprintf('iteration =  %d: , 1+2+3: %.4f, obj: %.4f; err: %.4f  \n', ...
        iter, Term1+Term2+Term3, obj(iter), err);
end

