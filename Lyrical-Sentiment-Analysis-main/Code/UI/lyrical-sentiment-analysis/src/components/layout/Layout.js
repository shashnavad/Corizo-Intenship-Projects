import NavBar from "./NavBar";
import Footer from "./Footer";

import css from "./Layout.module.css";
	
function Layout(props) {
	return (
		<div className={css.pageContainer}>
			<div className={css.contentWrap}>
				<NavBar />
				<main>{props.children}</main>
				<div ></div>
			</div>
			<Footer />
		</div>
	);
}

export default Layout;