import css from "./Partition.module.css";

function Partition(props) {
	return <div className={css.partition}>{props.children}</div>;
}

export default Partition;