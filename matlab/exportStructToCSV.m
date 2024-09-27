% exportStructToCSV(Brain.LH.A2009,'LHA2009.csv');
% exportStructToCSV(Brain.RH.A2009,'RHA2009.csv');
% exportStructToCSV(Brain.LH.APARC,'LHAPARC.csv');
% exportStructToCSV(Brain.RH.APARC,'RHAPARC.csv');
% exportStructToCSV(Brain.LH.DKT,'LHDKT.csv');
% exportStructToCSV(Brain.RH.DKT,'RHDKT.csv');
% exportStructToCSV(Brain.Brain,'BRAIN.csv');
% exportStructToCSV(Brain.WM,'WM.csv');
% exportStructToCSV(Brain.ASEG,'ASEG.csv');
% exportStructToCSV(Brain.ASEG,'ASEG.csv');


function exportStructToCSV(structData, fileName)
% Otwórz plik do zapisu
fid = fopen(fileName, 'w');

% Pobierz wszystkie pola na najwy¿szym poziomie
topFields = fieldnames(structData);

% Pobierz etykiety kolumn i wartoœci
labels = structData.Labels;
values = structData.Val;

% Zapisz etykiety kolumn do pliku CSV
fprintf(fid, '%s,', labels{:});
fprintf(fid, '\n');

% Zapisz wartoœci do pliku CSV
[rows, cols] = size(values);
for r = 1:rows
fprintf(fid, '%g,', values(r, :));
fprintf(fid, '\n');
end

% Zamknij plik
fclose(fid);
end
