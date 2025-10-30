\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}

\LoadClass[manuscript,nonacm]{acmart}


% Basic packages
\RequirePackage{graphicx}
\RequirePackage{bookmark}
\RequirePackage{microtype}


% Very light watermark
\RequirePackage{draftwatermark}
\SetWatermarkAngle{45}
\SetWatermarkLightness{0.92}
\SetWatermarkScale{0.2}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ INFO ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\projecttitle}{Title}
\newcommand{\projectsubtitle}{Subtitle}
\newcommand{\name}{Lawrence Granda Zarzuela}
\newcommand{\useremail}{lg626@cornell.edu}
\newcommand{\github}{https://github.com/lawrencegranda}
\newcommand{\bibfile}{bibfile}
\newcommand{\coursename}{Cornell University. BIONB 3500}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ HEADER AND FOOTER ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers and footers are handled by the ACM class


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%------ TITLE ------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title is produced by ACM's \maketitle


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
	% Title and subtitle
	\title[\projecttitle]{\projecttitle}
	\ifx\projectsubtitle\empty\else
		\subtitle{\projectsubtitle}
	\fi

	% Author metadata with course affiliation
	\author{\name}
	\email{\useremail}
	\affiliation{%
		\institution{\coursename}
		\country{USA}}
	\authornote{GitHub: \url{\github}.}

	% Watermark content
	\SetWatermarkText{\name\ ~\textlangle{}\useremail\textrangle{}}

	\maketitle
}
