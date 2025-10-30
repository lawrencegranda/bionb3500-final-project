\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}

\LoadClass[12pt]{article}

% Basic packages
\RequirePackage{parskip}
\RequirePackage{cite}
\RequirePackage{graphicx}
\RequirePackage[hidelinks]{hyperref}

% Fonts
\RequirePackage{fontspec}
\defaultfontfeatures[EBGaramond]{
	Ligatures={Common},
	UprightFont=*-Regular,
	BoldFont=*-SemiBold,
	ItalicFont=*-Italic,
	BoldItalicFont=*-SemiBoldItalic,
	Extension=.otf}

\setmainfont{EBGaramond}

% Page layout
\RequirePackage[margin=1in]{geometry}
\geometry{
	letterpaper,
	top=1in,
	left=1in,
	right=1in,
	bottom=1in,
	headheight=14pt,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ INFO ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\projecttitle}{Title}
\newcommand{\projectsubtitle}{Subtitle}
\newcommand{\name}{Lawrence Granda Zarzuela}
\newcommand{\email}{lg626@cornell.edu}
\newcommand{\github}{https://github.com/lawrencegranda}
\newcommand{\bibfile}{bibfile}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ HEADER AND FOOTER ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\projecttitle}
\fancyhead[R]{\name}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ TITLE ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\makedocumenttitle}{
	\begin{center}
		\vspace*{-20pt}
		\textbf{\Large \projecttitle}

		\textbf{\large \projectsubtitle}
	\end{center}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ CITATIONS ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\makecitations}{
	\begingroup
	\renewcommand{\section}[2]{}%
	\bibliographystyle{acm}
	\bibliography{\bibfile}
	\endgroup
}

\newcommand{\makereferences}{
	\section*{References}
	\makecitations
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ AT BEGIN DOCUMENT ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\AtBeginDocument{
	\makedocumenttitle
}
