%___________________________________________________________________%
%  Multi-Objective Dragonfly Algorithm (MODA) source codes demo     %
%                           version 1.0                             %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper:                                                     %
%                                                                   %
%   S. Mirjalili, Dragonfly algorithm: a new meta-heuristic         %
%   optimization technique for solving single-objective, discrete,  %
%   and multi-objective problems, Neural Computing and Applications %
%   DOI: http://dx.doi.org/10.1007/s00521-015-1920-1                %
%___________________________________________________________________%

% clc;
% clear;
% close all;
% 
% % Change these details with respect to your problem%%%%%%%%%%%%%%
% ObjectiveFunction=@ZDT1;
% dim=5;
% lb=0;
% ub=1;
% m=2;
% 
% if size(ub,2)==1
%     ub=ones(1,dim)*ub;
%     lb=ones(1,dim)*lb;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
clc;

%%%% 设置实验参数范围，包括实验次数和实验函数范围
Num_Test=5;   %%%% 每个函数独立进行Num_Test轮?
Num_Experiment=30;   %%%% 函数是从F1-FNum_Functions
AlgorithmName='MODA'; %%% 控制函数名?

% Initial parameters of the MODA algorithm
max_iter=100;
N=100;
ArchiveMaxSize=100;

m = 5;   %目标维数

ALLFunction_AllTest=[];

% for ff=[1:21];
% for ff=[10:21];
% for ff=[1:4,6:9];
for ff=[7];
    clearvars -except Num_Test Num_Experiment AlgorithmName ALLFunction_AllTest ff max_iter N ArchiveMaxSize m AllTest_Results  problem_name 
    %%%%% 创建文件夹?
    string_0ALL=['000\',AlgorithmName,'_5维目标800次迭代100种群实验20210923\'];
    dirname00=[string_0ALL,'\F',num2str(ff),'\'];
   display(['**********  ',AlgorithmName,'算法优化F',num2str(ff),'的 ', 'M',num2str(m), ' 维实验   **********']);
   for testi=1:Num_Test   %%%% 控制每次实验测试次数
       dirname0=[dirname00,'test',num2str(testi),'_F',num2str(ff)];
       system(['mkdir ' dirname0]) %创建主文件夹
       dirname1=[dirname0,'\F',num2str(ff),'_fig'];
       system(['mkdir ' dirname1]) %创建文件夹  等待保存实验图像
       dirname2=[dirname0,'\F',num2str(ff),'_data'];
       system(['mkdir ' dirname2]) %创建文件夹  等待保存实验图像
       for kk=1:30 %%%% 控制实验次数的循环
           display(['**********  ',AlgorithmName,'算法优化F',num2str(ff),'的  第  ', num2str(kk), ' 次实验   **********']);
            rand('state',sum(100*clock));
            problem_name=['F',num2str(ff)];
            [ ub,lb,dim ] = generate_boundary1( problem_name,m );%Upper and Lower Bound of Decision Variables  %%%生成决策空间中变量上界、下界和维度
            
            tic;  % CPU time measure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Archive_X=zeros(100,dim);
Archive_F=ones(100,m)*inf;

Archive_member_no=0;

r=(ub-lb)/2;
V_max=(ub(1)-lb(1))/10;

Food_fitness=inf*ones(1,m);
Food_pos=zeros(dim,1);

Enemy_fitness=-inf*ones(1,m);
Enemy_pos=zeros(dim,1);

X=initialization(N,dim,ub,lb);
fitness=zeros(N,2);

DeltaX=initialization(N,dim,ub,lb);
iter=0;

position_history=zeros(N,max_iter,dim);

for iter=1:max_iter
    
    r=(ub-lb)/4+((ub-lb)*(iter/max_iter)*2);
    
    w=0.9-iter*((0.9-0.2)/max_iter);
    
    my_c=0.1-iter*((0.1-0)/(max_iter/2));
    if my_c<0
        my_c=0;
    end
    
    if iter<(3*max_iter/4)
        s=my_c;             % Seperation weight
        a=my_c;             % Alignment weight
        c=my_c;             % Cohesion weight
        f=2*rand;           % Food attraction weight
        e=my_c;             % Enemy distraction weight
    else
        s=my_c/iter;        % Seperation weight
        a=my_c/iter;        % Alignment weight
        c=my_c/iter;        % Cohesion weight
        f=2*rand;           % Food attraction weight
        e=my_c/iter;        % Enemy distraction weight
    end
    
    for i=1:N %Calculate all the objective values first
%         Particles_F(i,:)=ObjectiveFunction(X(:,i)');
          Particles_F(i,:)=test_function(X(:,i)',dim,m,problem_name);
        if dominates(Particles_F(i,:),Food_fitness)
            Food_fitness=Particles_F(i,:);
            Food_pos=X(:,i);
        end
        
        if dominates(Enemy_fitness,Particles_F(i,:))
            if all(X(:,i)<ub') && all( X(:,i)>lb')
                Enemy_fitness=Particles_F(i,:);
                Enemy_pos=X(:,i);
            end
        end
    end
    
    [Archive_X, Archive_F, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, X, Particles_F, Archive_member_no);
    
    if Archive_member_no>ArchiveMaxSize
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, m);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no]=HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, m);
    end
    
    Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, m);
    
    % Chose the archive member in the least population area as foods
    % to improve coverage
    index=RouletteWheelSelection(1./Archive_mem_ranks);
    if index==-1
        index=1;
    end
    Food_fitness=Archive_F(index,:);
    Food_pos=Archive_X(index,:)';
       
    % Chose the archive member in the most population area as enemies
    % to improve coverage
    index=RouletteWheelSelection(Archive_mem_ranks);
    if index==-1
        index=1;
    end
    Enemy_fitness=Archive_F(index,:);
    Enemy_pos=Archive_X(index,:)';
    
    for i=1:N
        index=0;
        neighbours_no=0;
        
        clear Neighbours_V
        clear Neighbours_X
        % Find the neighbouring solutions
        for j=1:N
            Dist=distance(X(:,i),X(:,j));
            if (all(Dist<=r) && all(Dist~=0))
                index=index+1;
                neighbours_no=neighbours_no+1;
                Neighbours_V(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end
        
        % Seperation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.1)
        S=zeros(dim,1);
        if neighbours_no>1
            for k=1:neighbours_no
                S=S+(Neighbours_X(:,k)-X(:,i));
            end
            S=-S;
        else
            S=zeros(dim,1);
        end
        
        % Alignment%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.2)
        if neighbours_no>1
            A=(sum(Neighbours_V')')/neighbours_no;
        else
            A=DeltaX(:,i);
        end
        
        % Cohesion%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.3)
        if neighbours_no>1
            C_temp=(sum(Neighbours_X')')/neighbours_no;
        else
            C_temp=X(:,i);
        end
        
        C=C_temp-X(:,i);
        
        % Attraction to food%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.4)
        Dist2Attraction=distance(X(:,i),Food_pos(:,1));
        if all(Dist2Attraction<=r)
            F=Food_pos-X(:,i);
            iter;
        else
            F=0;
        end
        
        % Distraction from enemy%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.5)
        Dist=distance(X(:,i),Enemy_pos(:,1));
        if all(Dist<=r)
            E=Enemy_pos+X(:,i);
        else
            E=zeros(dim,1);
        end
        
        for tt=1:dim
            if X(tt,i)>ub(tt)
                X(tt,i)=lb(tt);
                DeltaX(tt,i)=rand;
            end
            if X(tt,i)<lb(tt)
                X(tt,i)=ub(tt);
                DeltaX(tt,i)=rand;
            end
        end
        
        
        if any(Dist2Attraction>r)
            if neighbours_no>1
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,i)>V_max
                        DeltaX(j,i)=V_max;
                    end
                    if DeltaX(j,i)<-V_max
                        DeltaX(j,i)=-V_max;
                    end
                    X(j,i)=X(j,i)+DeltaX(j,i);
                end
                
            else
                X(:,i)=X(:,i)+Levy(dim)'.*X(:,i);
                DeltaX(:,i)=0;
            end
        else    
            for j=1:dim
                DeltaX(j,i)=s*S(j,1)+a*A(j,1)+c*C(j,1)+f*F(j,1)+e*E(j,1) + w*DeltaX(j,i);
                if DeltaX(j,i)>V_max
                    DeltaX(j,i)=V_max;
                end
                if DeltaX(j,i)<-V_max
                    DeltaX(j,i)=-V_max;
                end
                X(j,i)=X(j,i)+DeltaX(j,i);
            end
        end
        
        Flag4ub=X(:,i)>ub';
        Flag4lb=X(:,i)<lb';
        X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
        
    end
    
%     display(['At the iteration ', num2str(iter), ' there are ', num2str(Archive_member_no), ' non-dominated solutions in the archive']);
    HisPF{iter} = Archive_F;
end


% figure
% 
% Draw_ZDT1();
% 
% hold on
% if obj_no==2
%     plot(Archive_F(:,1),Archive_F(:,2),'ko','MarkerSize',8,'markerfacecolor','k');
% else
%     plot3(Archive_F(:,1),Archive_F(:,2),Archive_F(:,3),'ko','MarkerSize',8,'markerfacecolor','k');
% end
% legend('True PF','Obtained PF');
% title('MODA');

            time=toc;
            PF = Archive_F;
            cg_curve=HisPF; %%% 历史目标函数值
            Time(kk)=time;
            cc=strcat(dirname2,'\',AlgorithmName,'优化次数_',num2str(kk),'.mat');
            result.time=time;
           
            true_PF=TPF(m,ArchiveMaxSize, problem_name);

         %%% using the matlab codes for calculating metric values
         

            hv = HV(PF,true_PF);   %超体积?
            gd = GD(PF, true_PF);                       %世代距离

            sp = Spacing(PF, true_PF);                  %空间分布 
            igd = IGD(PF, true_PF);            %反向世代距离

            hvd(kk)=hv;
            gdd(kk)=gd;
            ssp(kk)=sp;
            igdd(kk)=igd;

             save(cc)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       end
       mean_IGD = mean(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD平均值 : ', num2str(mean_IGD)]);
       std_IGD=std(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD标准差 : ', num2str(std_IGD)]);
       max_IGD=max(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD最大值 : ', num2str(max_IGD)]);
       min_IGD=min(igdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的IGD最小值 : ', num2str(min_IGD)]);
       display('******************************** ');
       
       
       mean_GD = mean(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD平均值 : ', num2str(mean_GD)]);
       std_GD=std(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD标准差 : ', num2str(std_GD)]);
       max_GD=max(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD最大值 : ', num2str(max_GD)]);
       min_GD=min(gdd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的GD最小值 : ', num2str(min_GD)]);
       display('******************************** ');
      
       
       mean_HV = mean(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV平均值 : ', num2str(mean_HV)]);
       std_HV=std(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV标准差 : ', num2str(std_HV)]);
       max_HV=max(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV最大值 : ', num2str(max_HV)]);
       min_HV=min(hvd);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的HV最小值 : ', num2str(min_HV)]);
       display('******************************** ');
       
       mean_SP = mean(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP平均值 : ', num2str(mean_SP)]);
       std_SP=std(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP标准差 : ', num2str(std_SP)]);
       max_SP=max(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP最大值 : ', num2str(max_SP)]);
       min_SP=min(ssp);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的SP最小值 : ', num2str(min_SP)]);
       display('******************************** ');
       
       mean_time=mean(Time);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的运行时间平均值 : ', num2str(mean_time)]);
       std_time=std(Time);
       display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的运行时间标准差 : ', num2str(std_time)]);
       display('******************************** ');
%        mean_X=mean(Best_X);
%         display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的最优解平均值 : ', num2str(mean_X)]);
%         std_X=std(Best_X);
%         display([AlgorithmName,'_Functions_F',num2str(ff),'测试',num2str(kk),'次实验的最优解标准差 : ', num2str(std_X)]);
%         %%%%%%%%%%%%%%%%%%
        cd=strcat(dirname0,'\Result汇总结果.mat');
        Result.IGDmean=mean_IGD;
        Result.IGDstd=std_IGD;
        Result.IGDmax=max_IGD;
        Result.IGDmin=min_IGD;
      
        
        Result.GDmean=mean_GD;
        Result.GDstd=std_GD;
        Result.GDmax=max_GD;
        Result.GDmin=min_GD;
        
        Result.HVmean=mean_HV;
        Result.HVstd=std_HV;
        Result.HVmax=max_HV;
        Result.HVmin=min_HV;
        
        Result.SPmean=mean_SP;
        Result.SPstd=std_SP;
        Result.SPmax=max_SP;
        Result.SPmin=min_SP;
        
        Result.Tmean=mean_time;
        Result.Tstd=std_time;
        %         Result.Xmean=mean_X;
        %         Result.Xstd=std_X;
        %         Result.Best_Y=Best_Y;
        %         Result.Best_X=Best_X;
        Result.Time=Time;
        %         Result.ResultVector=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_time,std_time];
        Result.ResultVector=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_HV,std_HV,max_HV,min_HV,mean_SP,std_SP,max_SP,min_SP,mean_time,std_time];
        %         Result.Best_History_Y=History_Y;
        save(cd,'Result')
        
        
        %         AllTest_Results(testi,:)=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_time,std_time];
        % AllTest_Results(testi,:)=[mean_IGD,std_IGD,max_IGD,min_IGD,mean_GD,std_GD,max_GD,min_GD,mean_time,std_time];
        AllTest_Results(testi,:)=[mean_IGD,std_IGD,mean_GD,std_GD,mean_HV,std_HV,mean_SP,std_SP,mean_time,std_time];
    end
    cd=strcat(dirname00,'Result_AllTest.mat');
    save(cd,'AllTest_Results')
    ALLFunction_AllTest=[ALLFunction_AllTest;AllTest_Results];
end


cd=strcat(string_0ALL,'ALLFunction_AllTest.mat');
save(cd,'ALLFunction_AllTest')