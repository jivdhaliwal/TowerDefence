package towerdefence.engine.pathfinding;

/**
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */

import org.newdawn.slick.util.pathfinding.Mover;

public class UnitMover implements Mover {

    // ID of unit moving
    private int type;

    public UnitMover(int type) {
        this.type = type;
    }

    /*
     * Get ID of moving unit
     */
    public int getType() {
        return type;
    }

}
