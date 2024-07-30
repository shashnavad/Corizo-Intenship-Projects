import { useContext } from "react";
import StateContext from "../store/state-context";

import css from "./MainPage.module.css";
import Partition from "../components/layout/Partition";
import Lyrics from "../components/partitions/Lyrics";
import DeepLearning from "../components/partitions/DeepLearning";
import Classical from "../components/partitions/Classical";

function MainPage() {
	const stateContext = useContext(StateContext); 
	
	if (stateContext.results !== null) {
		if (stateContext.model === "Classical") {
			return (
				<div className={css.mainpage}>
					<Partition><Lyrics /></Partition>
					<Partition><Classical /></Partition>
				</div>
			);
		}

		if (stateContext.model === "Deep Learning") {
			return (
				<div className={css.mainpage}>
					<Partition><Lyrics /></Partition>
					<Partition><DeepLearning /></Partition>
				</div>
			);
		}
	}
	
	return (
		<div className={css.mainpage}>
			<Partition><Lyrics /></Partition>
		</div>
	);
}

export default MainPage;