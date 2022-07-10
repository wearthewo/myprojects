
Select *
FROM portfolioproject..CovidDeaths
where continent is NOT NULL
order by 3,4


--Select * 
--FROM portfolioproject..CovidVaccs


Select location,date,total_cases,new_cases,total_deaths,population
FROM portfolioproject..CovidDeaths
ORDER BY 1,2


-- Death percentage in Greece
Select location,date,total_cases,total_deaths,(total_deaths/total_cases)*100 as DeathPercentage
FROM portfolioproject..CovidDeaths
WHERE location = 'Greece'
ORDER BY 1,2


-- total cases / population percentage in Greece
-- Covid percentge
Select location,date,total_cases,population,(total_cases/population)*100 as CovidPercentage
FROM portfolioproject..CovidDeaths
WHERE location = 'Greece'
ORDER BY 1,2


--Countries with highest infection rate
Select location,MAX(total_cases) as HighestInfectionCount,population,(MAX(total_cases)/population)*100 as HighestInfectionRrate
FROM portfolioproject..CovidDeaths
GROUP BY location,Population
ORDER BY HighestInfectionRrate desc


-- HighestDeathCount per Country
Select location,MAX(CAST(total_deaths AS INT)) as HighestDeathCount
FROM portfolioproject..CovidDeaths
where continent is NOT NULL
GROUP BY location
ORDER BY HighestDeathCount desc

-- HighestDeathCount per Continent
Select location,MAX(CAST(total_deaths AS INT)) as HighestDeathCount
FROM portfolioproject..CovidDeaths
where continent is  NULL
GROUP BY location
ORDER BY HighestDeathCount desc

-- HighestDeathCount per Continent
Select continent,MAX(CAST(total_deaths AS INT)) as HighestDeathCount
FROM portfolioproject..CovidDeaths
where continent is not NULL
GROUP BY continent
ORDER BY HighestDeathCount desc


--NUMBERS ALL over the word with sums for all the days

Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, SUM(cast(new_deaths as int))/SUM(New_Cases)*100 as DeathPercentage
From portfolioproject..CovidDeaths
--Where location like '%states%'
where continent is not null 
--Group By date
order by 1,2

--JOINS BETWEEN DEATHS AND VACCS
-- PARTITION BY FOR ROLLINGSUMS

Select dead.continent,dead.location,dead.date,dead.population,vacc.new_vaccinations,SUM(CONVERT(int,vacc.new_vaccinations)) OVER (Partition by dead.Location Order by dead.location, dead.Date) as RollingPeopleVaccinated
FROM portfolioproject..CovidDeaths dead
JOIN portfolioproject..CovidVaccs vacc
	ON dead.location=vacc.location
	AND dead.date=vacc.date
Where dead.continent is not NULL
order by 2,3




With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dead.continent, dead.location, dead.date, dead.population, vacc.new_vaccinations
, SUM(CONVERT(int,vacc.new_vaccinations)) OVER (Partition by dead.Location Order by dead.location, dead.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From PortfolioProject..CovidDeaths dead
Join PortfolioProject..CovidVaccs vacc
	On dead.location = vacc.location
	and dead.date = vacc.date
where dead.continent is not null 
--order by 2,3
)
Select *, (RollingPeopleVaccinated/Population)*100
From PopvsVac

--TEMP TABLE

DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)

Insert into #PercentPopulationVaccinated
Select dead.continent, dead.location, dead.date, dead.population, vacc.new_vaccinations
, SUM(CONVERT(int,vacc.new_vaccinations)) OVER (Partition by dead.Location Order by dead.location, dead.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From PortfolioProject..CovidDeaths dead
Join PortfolioProject..CovidVaccs vacc
	On dead.location = vacc.location
	and dead.date = vacc.date
--where dea.continent is not null 
--order by 2,3

Select *, (RollingPeopleVaccinated/Population)*100
From #PercentPopulationVaccinated




-- Creating View 

Create View PercentPopulationVaccinated as
Select dead.continent, dead.location, dead.date, dead.population, vacc.new_vaccinations
, SUM(CONVERT(int,vacc.new_vaccinations)) OVER (Partition by dead.Location Order by dead.location, dead.Date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/population)*100
From PortfolioProject..CovidDeaths dead
Join PortfolioProject..CovidVaccs vacc
	On dead.location = vacc.location
	and dead.date = vacc.date
where dead.continent is not null 




