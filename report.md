# Filmbaserat rekommendationssystem

## Introduktion / Problemställning
Syftet var att konstruera ett system som, givet en film som indata, returnerar fem relevanta filmrekommendationer. Problemet kan formuleras som att uppskatta likhet mellan filmer utifrån tillgänglig metadata i MovieLens-datamängden. För uppgiften var det centralt att rekommendationerna kunde genereras på ett stabilt sätt utan att hela `ratings.csv` behövde laddas i minnet, eftersom den filen är mycket stor.

Den teoretiska grund som användes var innehållsbaserad rekommendation med textrepresentation. Varje film representerades som ett dokument bestående av genreord och användargenererade taggar. Dokumenten transformerades därefter med TF-IDF, där vikten för ett ord ökade när ordet var vanligt i en film men ovanligt i hela datamängden. För ett ord $t$ i ett dokument $d$ användes principen
`TF-IDF(t, d) = TF(t, d) * IDF(t)`
där `IDF(t)` växer när ordet förekommer i färre dokument. Likhet mellan två filmer beräknades sedan med cosinuslikhet,
`cos(A, B) = (A · B) / (||A|| ||B||)`
vilket är lämpligt för glesa textvektorer eftersom metoden jämför riktning snarare än absolut storlek. Valet låg också i linje med kursens arbetssätt, där databehandling, visualisering och klassiska metoder i `scikit-learn` har varit centrala. Collaborative filtering studerades teoretiskt, men användes inte som slutlig modell eftersom användar-film-matrisen blev för minneskrävande i förhållande till uppgiftens omfattning.

## Data-analys (EDA)
Den slutliga modellen byggdes på `movies.csv` och `tags.csv`. `movies.csv` innehöll `movieId`, filmtitel och genrelista. Genreinformationen var separerad med `|`, vilket gjorde det naturligt att dela upp varje genre till separata textuella attribut. Analysen visade att vissa genrer, särskilt drama och komedi, förekom betydligt oftare än övriga. Detta var relevant eftersom en ren genrebaserad modell riskerade att ge alltför generella rekommendationer.

`tags.csv` innehöll fria nyckelord skapade av användare, exempelvis termer som beskrev ton, tema, miljö eller stil. Dessa taggar gav mer finmaskig information än genrerna och förbättrade möjligheten att skilja mellan filmer inom samma genre. EDA:n visade också att många filmer saknade rik tagginformation, medan andra hade ett stort antal återkommande taggar. Därför grupperades taggar per film och slogs samman till en gemensam textrepresentation.

`ratings.csv` analyserades översiktligt men användes inte i slutmodellen. Den huvudsakliga observationen var att filen var mycket stor och att en full användning hade krävt en användar-film-matris med mycket hög dimensionalitet. För denna uppgift bedömdes därför att informationen i genre och taggar gav bättre balans mellan kvalitet, tydlighet och beräkningskostnad. EDA-delen genomfördes i samma anda som övriga kursmoment: datamängden lästes in, centrala fördelningar inspekterades och endast de samband som påverkade modellvalet togs vidare.

## Modell
Flera modellidéer jämfördes på konceptuell nivå: en enkel genrebaserad likhetsmodell, en innehållsbaserad modell med TF-IDF, samt en kollaborativ modell baserad på användarbetyg. Den modell som valdes i slutändan var den innehållsbaserade modellen med kombinerade genre- och taggdata, eftersom den gav tydliga och rimliga rekommendationer utan stora minneskrav.

För varje film skapades ett textfält där genrer och sammanslagna taggar kombinerades. Därefter användes `TfidfVectorizer` med engelska stoppord borttagna. I implementationen användes även begränsning av vokabulärstorlek och filtrering av mycket ovanliga respektive mycket vanliga termer för att minska brus. Den centrala modellen var alltså en TF-IDF-representation följd av cosinuslikhet mellan den valda filmen och samtliga övriga filmer. Endast de fem högst rankade filmerna, bortsett från inmatningsfilmen själv, returnerades. Arbetsgången i repot följde därmed en tydlig pipeline: nedladdning av `ml-latest`, explorativ analys i `data_analysis.py`, konstruktion av metadata i `recommendation_system.py`, samt testning av rekommendationer både via kommandorad och via ett enklare Dash-gränssnitt i `app.py`.

En viktig parameter var hur metadata byggdes upp. Genrer transformerades från formatet `Action|Adventure|Sci-Fi` till separata ord, medan taggar grupperades per `movieId` och slogs samman till en sammanhängande text. Denna kombination gav bättre resultat än att använda enbart genre, eftersom taggarna fångade mer specifika egenskaper som stämning, tema och stil. Möjligheten att använda dimensionsreduktion eller klustring identifierades också, vilket harmonierar med kursens moment om osuperviserat lärande, men dessa tekniker användes inte i den slutliga modellen.

## Resultat
Den valda modellen producerade i flera fall rekommendationer som var intuitivt rimliga. När filmer med tydliga teman eller väletablerade filmserier användes som indata gav modellen rekommendationer med hög semantisk närhet. Exempelvis gav familje- och animationsfilmer rekommendationer från närliggande serier eller filmer med liknande ton. För science fiction-filmer observerades rekommendationer där både genre och centrala taggar, såsom futuristiska eller mörka teman, återkom.

Resultatet utvärderades huvudsakligen kvalitativt, eftersom uppgiften fokuserade på att bygga ett fungerande rekommendationssystem snarare än att optimera ett specifikt numeriskt mått. Den slutliga modellen bedömdes som bättre än en ren genrebaserad lösning, eftersom rekommendationerna blev mer specifika. Jämfört med en full kollaborativ modell var lösningen enklare att köra på vanlig hårdvara och mer robust i denna begränsade implementation. Resultaten stöddes dessutom av den explorativa analysen och de testkörningar som genomfördes i repot.

Ett representativt exempel på programutskrift var följande:
```text
Input: The Matrix
1. Matrix Revolutions, The (2003)
2. Matrix Reloaded, The (2003)
3. Terminator 2: Judgment Day (1991)
4. Star Wars: Episode V - The Empire Strikes Back (1980)
5. 2001: A Space Odyssey (1968)
```
Detta resultat visar att modellen kunde fånga både direkta franchise-samband och mer övergripande innehållslikhet inom science fiction.

## Diskussion
Den främsta begränsningen var att slutmodellen var innehållsbaserad och därmed beroende av kvaliteten i genrer och taggar. Om en film hade få eller oinformativa taggar försämrades rekommendationernas kvalitet. Modellen hade också en tendens att rekommendera filmer som låg mycket nära inmatningsfilmen, vilket förbättrade träffsäkerheten men minskade variationen.

En annan begränsning var att användarbeteenden i `ratings.csv` inte utnyttjades i slutmodellen. Därmed kunde inte samband av typen “användare som gillade denna film gillade också ...” fångas fullt ut. Samtidigt innebar detta val att lösningen blev betydligt enklare att förklara, snabbare att köra och bättre anpassad till tillgängliga resurser. För problemställningen innebar resultatet att en relativt enkel textbaserad modell var tillräcklig för att generera trovärdiga rekommendationer på ett praktiskt sätt.

Vid fortsatt arbete hade en naturlig förbättring varit att kombinera den nuvarande innehållsbaserade modellen med en lättviktsvariant av betygsbaserad rangordning, alternativt att använda dimensionsreduktion eller klustring för att öka variationen bland rekommendationerna. För den aktuella uppgiften bedömdes dock den valda modellen ge en god balans mellan begriplighet, prestanda och rekommendationskvalitet.
