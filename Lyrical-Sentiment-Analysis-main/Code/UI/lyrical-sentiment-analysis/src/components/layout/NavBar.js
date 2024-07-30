import css from "./NavBar.module.css";

function NavBar() {
	return (
		<header className={css.header}>
			<div className={css.title}>Lyrical Sentiment Analysis</div>
		</header>
	);
}

export default NavBar;